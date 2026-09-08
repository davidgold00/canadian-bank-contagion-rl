"""Verify identity and every route, optionally through Vercel protection bypass."""
import argparse,json,os,subprocess
from urllib.request import Request,urlopen
from urllib.parse import urljoin

def verify(base,expected,vercel_cli=False):
    headers={'Cache-Control':'no-cache'}
    if os.environ.get('VERCEL_AUTOMATION_BYPASS_SECRET'):headers['x-vercel-protection-bypass']=os.environ['VERCEL_AUTOMATION_BYPASS_SECRET']
    def get(path):
        if vercel_cli:
            result=subprocess.run(
                ['vercel','curl',path,'--deployment',base,'--','--location','--silent','--fail'],
                check=True,capture_output=True,
            )
            return result.stdout
        with urlopen(Request(urljoin(base.rstrip('/')+'/',path.lstrip('/')),headers=headers),timeout=30) as r:
            if r.status!=200:raise RuntimeError(f'HTTP {r.status}: {path}')
            return r.read()
    manifest=json.loads(get('/snapshot-manifest.json'))
    if manifest.get('snapshot_id')!=expected:raise RuntimeError(f"Expected {expected}; deployment contains {manifest.get('snapshot_id')}")
    for slug in ['','risk','scenarios','models','decision','performance','research','review']:
        for prefix in ['/',manifest['case_url']]:
            html=get(prefix+slug).decode()
            if f'data-snapshot="{expected}"' not in html or 'Check for updated data' not in html:raise RuntimeError('Wrong case or incomplete route: '+prefix+slug)
    for file in ['case.json','assets/update.js','assets/scenario.js','assets/review.css','review.md','methodology.md','model-card.md']:
        get(manifest['case_url']+file)
    print(f'Verified {expected}: all root/pinned routes and required evidence assets')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--case',default='artifacts/current/case.json');p.add_argument('--vercel-cli',action='store_true');a=p.parse_args()
    verify(a.url,json.load(open(a.case))['manifest']['snapshot_id'],a.vercel_cli)
