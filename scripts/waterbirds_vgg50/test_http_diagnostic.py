import io
import unittest
import urllib.error
import urllib.request
from scripts.waterbirds_vgg50.diagnose_http import diagnose, StopRedirect, redirect_url


class Response(io.BytesIO):
    def __init__(self,status,headers,body=b'',url='https://storage.invalid/object'):
        super().__init__(body);self.status=status;self.headers=headers;self.url=url;self.read_sizes=[]
    def read(self,n=-1):
        self.read_sizes.append(n)
        if n!=1:raise AssertionError('Diagnostic may read only one byte')
        return super().read(n)
    def geturl(self):return self.url


class Tests(unittest.TestCase):
    def test_stock_redirect_retains_get_range_headers_and_changes_head(self):
        handler=urllib.request.HTTPRedirectHandler()
        headers={'Range':'bytes=0-0','Accept-Encoding':'identity','User-Agent':'test-agent'}
        for status in (301,302,303,307,308):
            original=urllib.request.Request('https://caltech.invalid/file',method='GET',headers=headers)
            forwarded=handler.redirect_request(original,None,status,'redirect',{},'https://storage.invalid/file?sig=%2F%2B')
            self.assertEqual(forwarded.get_method(),'GET')
            self.assertEqual(forwarded.get_header('Range'),'bytes=0-0')
            self.assertEqual(forwarded.get_header('Accept-encoding'),'identity')
            self.assertEqual(forwarded.get_header('User-agent'),'test-agent')
        head=urllib.request.Request('https://caltech.invalid/file',method='HEAD')
        # Evidence of a separate stdlib behavior, not a claimed cause of the GET403.
        forwarded=handler.redirect_request(head,None,302,'redirect',{},'https://storage.invalid/file')
        self.assertEqual(forwarded.get_method(),'GET')

    def test_multiple_redirects_read_one_byte_and_preserve_headers(self):
        replies=[Response(302,{'Location':'https://storage.invalid/a?sig=%2F%2B'},b'ignored redirect body'),
                 Response(307,{'Location':'/b?sig=%2F%2B'},b'ignored'),
                 Response(206,{'Content-Range':'bytes 0-0/1000','Content-Length':'1'},b'x')]
        seen=[]
        def opener(request,timeout):
            seen.append(request)
            for req in seen:
                self.assertEqual(req.get_method(),'GET')
                self.assertEqual(req.get_header('Range'),'bytes=0-0')
                self.assertEqual(req.get_header('Accept-encoding'),'identity')
                self.assertEqual(req.get_header('User-agent'),'curl/fixture')
            return replies[len(seen)-1]
        result=diagnose('https://caltech.invalid/file',user_agent='curl/fixture',opener=opener)
        self.assertTrue(result['success']);self.assertEqual(result['body_bytes_read'],1)
        self.assertEqual([r.read_sizes for r in replies],[[],[],[1]])
        self.assertTrue(all(r.closed for r in replies))
        self.assertTrue(seen[-1].full_url.endswith('?sig=%2F%2B'))
        self.assertNotIn('sig=%2F%2B',str(result)) # Query values are not logged.

    def test_errors_and_200_do_not_read_body(self):
        for status,headers in [(200,{}),(403,{}),(206,{'Content-Range':'bytes 0-0/100','Content-Length':'2'}),
                               (206,{'Content-Range':'bytes 1-1/100','Content-Length':'1'})]:
            with self.subTest(status=status,headers=headers):
                response=Response(status,headers,b'potentially large body')
                result=diagnose('https://caltech.invalid/file',opener=lambda *a,**k:response)
                self.assertFalse(result['success']);self.assertEqual(result['body_bytes_read'],0)
                self.assertEqual(response.read_sizes,[]);self.assertTrue(response.closed)
        response=Response(403,{},b'forbidden')
        def fail(*args,**kwargs):
            raise urllib.error.HTTPError('https://storage.invalid/file',403,'Forbidden',{},response)
        result=diagnose('https://caltech.invalid/file',opener=fail)
        self.assertFalse(result['success']);self.assertEqual(response.read_sizes,[]);self.assertTrue(response.closed)

    def test_stop_redirect_never_drains_and_url_safety(self):
        handler=StopRedirect();body=Response(302,{},b'large redirect body')
        with self.assertRaises(urllib.error.HTTPError) as raised:
            handler.http_error_302(urllib.request.Request('https://a.invalid'),body,302,'redirect',{'Location':'https://b.invalid'})
        raised.exception.close();self.assertEqual(body.read_sizes,[])
        url='https://storage.invalid/x?signature=a%2Fb%2Bc&part=1'
        self.assertEqual(redirect_url('https://a.invalid',url,'stdlib'),url)
        self.assertEqual(redirect_url('https://a.invalid',url,'literal'),url)
        for target in ('file:///tmp/x','http://storage.invalid/x','https://user:password@storage.invalid/x'):
            with self.assertRaises(ValueError):redirect_url('https://a.invalid',target,'stdlib')

if __name__=='__main__':unittest.main()
