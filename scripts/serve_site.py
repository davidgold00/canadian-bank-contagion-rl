"""Read-only local preview of the atomically published static release."""
import argparse,sys
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1]
class ReadOnlyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Resolve once per request; an in-flight request retains its complete release.
        self.directory=str(self.server.release_pointer.resolve())
        super().do_GET()
    def do_HEAD(self):
        self.directory=str(self.server.release_pointer.resolve());super().do_HEAD()
    def translate_path(self,path):
        translated=Path(super().translate_path(path))
        if not translated.exists() and translated.with_suffix('.html').is_file():translated=translated.with_suffix('.html')
        return str(translated)
    def do_POST(self):self.send_error(405,'Read-only preview; use the authenticated owner CLI/workflow')
    def end_headers(self):
        self.send_header('Cache-Control','no-store');super().end_headers()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);p.add_argument('--directory',default='build/current');a=p.parse_args()
    directory=(ROOT/a.directory).absolute()
    if not directory.exists():sys.exit('Render a case first with python scripts/render_release.py')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),ReadOnlyHandler);server.release_pointer=directory
    print(f'Read-only preview: http://127.0.0.1:{a.port}',flush=True);server.serve_forever()
