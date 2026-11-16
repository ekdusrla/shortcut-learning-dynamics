"""Explicit, one-byte Range GET diagnostics; no dataset downloads or HEAD calls.

Redirect/error/200 bodies are never read. Each invocation reads at most ONE
application body byte, only after final 206 headers pass validation. urllib and
OS/TLS buffers can prefetch network bytes, so this is not a wire-byte guarantee.
"""
import argparse
import hashlib
import json
import re
import string
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
REDIRECTS = (301, 302, 303, 307, 308)


class StopRedirect(urllib.request.HTTPRedirectHandler):
    # The standard handler drains fp.read() before following a redirect.
    # Return control without consuming ANY redirect body instead.
    def http_error_302(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)
    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def url_info(url):
    parsed = urllib.parse.urlsplit(url)
    return dict(scheme=parsed.scheme, host=parsed.hostname, path=parsed.path,
                query_keys=[p.split('=', 1)[0] for p in parsed.query.split('&') if p],
                query_sha256=hashlib.sha256(parsed.query.encode()).hexdigest(),
                full_url_sha256=hashlib.sha256(url.encode()).hexdigest())


def redirect_url(current, location, style):
    if style == 'stdlib':
        # Mirror Python 3.11 HTTPRedirectHandler URL normalization, but not its
        # body drain. GET remains GET; the script never issues HEAD.
        parts = urllib.parse.urlparse(location)
        if not parts.path and parts.netloc:
            parts = list(parts); parts[2] = '/'
            location = urllib.parse.urlunparse(parts)
        location = urllib.parse.quote(location, encoding='iso-8859-1', safe=string.punctuation)
    elif style != 'literal':
        raise ValueError('Unknown redirect style')
    target = urllib.parse.urljoin(current, location)
    parsed = urllib.parse.urlsplit(target)
    if parsed.scheme not in ('https', 'http') or parsed.username or parsed.password:
        raise ValueError('Redirect must be HTTP(S) without URL credentials')
    if urllib.parse.urlsplit(current).scheme == 'https' and parsed.scheme != 'https':
        raise ValueError('HTTPS downgrade refused')
    return target


def diagnose(url, user_agent=None, accept_encoding='identity', redirect_style='stdlib', no_proxy=False, opener=None):
    headers = {'Range':'bytes=0-0', 'User-Agent':user_agent or f'Python-urllib/{sys.version_info.major}.{sys.version_info.minor}'}
    if accept_encoding != 'implicit':
        headers['Accept-Encoding'] = '' if accept_encoding == 'empty' else accept_encoding
    if opener is None:
        handlers = [StopRedirect()]
        if no_proxy: handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers).open
    report = dict(python=sys.version, user_agent=headers['User-Agent'],
                  accept_encoding=accept_encoding, redirect_style=redirect_style,
                  proxy_disabled=no_proxy, proxy_schemes=sorted(urllib.request.getproxies()),
                  body_bytes_read=0, success=False, hops=[])
    current = url
    visited = set()
    for _ in range(8):
        if current in visited:
            report['error']='Redirect loop'; return report
        visited.add(current)
        request = urllib.request.Request(current, method='GET', headers=headers)
        hop = dict(request_url=url_info(current), method=request.get_method(),
                   request_headers=dict(request.header_items()),
                   implicit_accept_encoding='http.client sends identity if omitted' if accept_encoding=='implicit' else None)
        report['hops'].append(hop)
        try:
            response = opener(request, timeout=20)
        except urllib.error.HTTPError as error:
            response = error
        except (OSError, ValueError) as error:
            report['error']=str(error); return report
        try:
            code = response.code if isinstance(response, urllib.error.HTTPError) else response.status
            hop['status']=code
            hop['response_url']=url_info(response.geturl())
            fields=('Content-Range','Content-Length','Content-Encoding','ETag','Last-Modified','Date','Server','Via','Age','X-Cache','Content-Type')
            hop['response_headers']={key:response.headers.get(key) for key in fields}
            if code in REDIRECTS:
                location = response.headers.get('Location')
                if not location: raise ValueError('Redirect missing Location')
                hop['raw_location']=url_info(location)
                target = redirect_url(current, location, redirect_style)
                hop['next_url']=url_info(target)
                hop['range_forwarded']='bytes=0-0'
                current=target
                continue
            if code != 206:
                raise ValueError(f'HTTP {code}; body not read')
            if not re.fullmatch(r'bytes 0-0/[1-9][0-9]*', response.headers.get('Content-Range','')):
                raise ValueError('Invalid Content-Range; body not read')
            if response.headers.get('Content-Length')!='1' or response.headers.get('Content-Encoding','identity')!='identity':
                raise ValueError('Invalid length/encoding; body not read')
            payload=response.read(1)
            report['body_bytes_read']=len(payload)
            if len(payload)!=1: raise ValueError('Truncated one-byte response')
            report['success']=True
            return report
        except (OSError, ValueError) as error:
            report['error']=str(error); return report
        finally:
            response.close()
    report['error']='Too many redirects'; return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default=DEFAULT_URL)
    parser.add_argument('--user-agent',help='Default matches Python urllib; use the exact curl UA for comparison')
    parser.add_argument('--accept-encoding',choices=('identity','implicit','empty'),default='identity')
    parser.add_argument('--redirect-style',choices=('stdlib','literal'),default='stdlib')
    parser.add_argument('--no-proxy',action='store_true',help='Compare direct connection if permitted by server policy')
    args=parser.parse_args()
    parsed=urllib.parse.urlsplit(args.url)
    if parsed.scheme not in ('http','https') or parsed.username or parsed.password:
        parser.error('Use an HTTP(S) URL without credentials')
    result=diagnose(args.url,args.user_agent,args.accept_encoding,args.redirect_style,args.no_proxy)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if result['success'] else 1

if __name__=='__main__':raise SystemExit(main())
