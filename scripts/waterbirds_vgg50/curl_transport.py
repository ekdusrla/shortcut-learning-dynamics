"""curl HTTP adapter: argv only, no -L, no error/redirect body consumption.

A bounded OS pipe may prefetch body bytes; callers validate headers before read.
"""
import email.parser
import subprocess
import tempfile
import time
import sys
import urllib.parse
import urllib.request


class CurlTransportError(OSError):
    def __init__(self, code, host, detail):
        self.code=code;self.host=host
        super().__init__(f'curl exit {code}, host={host}: {detail}')


class RetryExhausted(CurlTransportError):
    def __init__(self,error,retries,operation):
        super().__init__(error.code,error.host,f'{operation}: retries={retries}, attempts={retries+1}, last_error_code={error.code}; {error}')
        self.retries=retries


class RetryPolicy:
    # HTTP/metadata/integrity failures and curl18/22/60 are not transient here.
    TRANSIENT={5,6,7,28,52,55,56}
    def __init__(self,max_retries=5,sleep=None):
        if not 0<=max_retries<=5:raise ValueError('Retry count must be 0..5')
        self.max_retries=max_retries;self.sleep=sleep or time.sleep
    def run(self,operation,call):
        for retry in range(self.max_retries+1):
            try:return call()
            except CurlTransportError as error:
                if isinstance(error,RetryExhausted) or error.code not in self.TRANSIENT:raise
                if retry==self.max_retries:raise RetryExhausted(error,retry,operation) from error
                delay=2**retry
                print(f'[network retry] host={error.host} operation={operation} retry={retry+1}/{self.max_retries} last_error_code={error.code} backoff={delay}s',file=sys.stderr)
                self.sleep(delay)


class CurlResponse:
    def __init__(self, request, timeout, executable='curl'):
        self.url=request.full_url; self.stderr=tempfile.TemporaryFile()
        argv=[executable,'--silent','--show-error','--no-buffer',
              '--include','--suppress-connect-headers','--max-redirs','0',
              '--connect-timeout','20','--max-time',str(timeout),'--proto','=http,https']
        if request.get_method()=='HEAD':argv.append('--head')
        elif request.get_method()!='GET':raise ValueError('Only HEAD/GET are allowed')
        for key,value in request.header_items():argv.extend(['--header',f'{key}: {value}'])
        argv.extend(['--url',self.url])
        try:self.process=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=self.stderr,shell=False)
        except BaseException:
            self.stderr.close();raise
        self.closed=False
        try:
            for _ in range(6):
                line=self.process.stdout.readline(8193)
                if not line:
                    self.check_exit()
                    raise OSError('curl returned no HTTP status')
                fields=line.decode('iso-8859-1').strip().split(' ',2)
                if len(fields)<2 or not fields[0].startswith('HTTP/') or not fields[1].isdigit():
                    raise OSError('Invalid curl HTTP status')
                self.status=int(fields[1]);size=len(line);lines=[]
                while True:
                    line=self.process.stdout.readline(8193);size+=len(line)
                    if not line:
                        self.check_exit();raise OSError('Truncated curl headers')
                    if size>65536:raise OSError('Oversized curl headers')
                    if line in (b'\r\n',b'\n'):break
                    lines.append(line)
                self.headers=email.parser.BytesParser().parsebytes(b''.join(lines)+b'\r\n')
                if self.status>=200:break
            else:raise OSError('Too many interim HTTP responses')
        except BaseException:
            self.close();raise

    def check_exit(self):
        code=self.process.wait(timeout=5)
        if code:
            self.stderr.seek(0)
            raise CurlTransportError(code,urllib.parse.urlsplit(self.url).hostname,self.stderr.read(4096).decode(errors='replace'))

    def read(self,n=-1):
        if n<0:raise ValueError('Unbounded curl body read refused')
        data=self.process.stdout.read(n)
        if len(data)<n:self.check_exit()
        return data

    def geturl(self):return self.url
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        if self.closed:return
        self.closed=True
        if self.process.poll() is None:self.process.terminate()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process.stdout.close();self.stderr.close()


class CurlOpener:
    def __init__(self,executable='curl',retry_policy=None):
        self.executable=executable;self.retry_policy=retry_policy or RetryPolicy()
    def __call__(self,request,timeout=60):
        current=request
        for _ in range(8):
            parsed=urllib.parse.urlsplit(current.full_url)
            if parsed.scheme not in ('http','https') or parsed.username or parsed.password:
                raise ValueError('Only HTTP(S) URLs without credentials allowed')
            response=CurlResponse(current,timeout,self.executable)
            if response.status not in (301,302,303,307,308):return response
            try:
                location=response.headers.get('Location')
                if not location:raise ValueError('Redirect missing Location')
                target=urllib.parse.urljoin(current.full_url,location)
                if parsed.scheme=='https' and urllib.parse.urlsplit(target).scheme!='https':
                    raise ValueError('HTTPS downgrade refused')
                headers={k:v for k,v in current.header_items() if k.lower() not in ('host','authorization','cookie','proxy-authorization')}
                current=urllib.request.Request(target,method=current.get_method(),headers=headers)
            finally:response.close() # Do not drain redirect bodies.
        raise OSError('Too many curl redirects')
