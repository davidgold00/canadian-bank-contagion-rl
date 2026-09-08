"""Reject stale/out-of-order promotion; intentional rollback is a separate operation."""
import argparse,json
from datetime import datetime
from urllib.request import Request,urlopen
from urllib.error import HTTPError

def may_publish(candidate,current):
    if candidate['snapshot_id']==current['snapshot_id']:return
    if candidate['feature_date']<current['feature_date']:raise ValueError('Refusing to publish an older data snapshot')
    if datetime.fromisoformat(candidate['generated_at'])<datetime.fromisoformat(current['generated_at']):raise ValueError('Refusing to publish an older prepared case after a newer one')

def guard(url,case):
    candidate=json.load(open(case))['manifest']
    try:
        with urlopen(Request(url.rstrip('/')+'/snapshot-manifest.json',headers={'Cache-Control':'no-cache'}),timeout=30) as r:current=json.load(r)
    except HTTPError as error:
        if error.code==404:return # First legacy→v2 transition.
        raise
    may_publish(candidate,current)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--case',default='artifacts/current/case.json');a=p.parse_args();guard(a.url,a.case)
