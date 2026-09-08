"""Restore immutable pinned cases from the published file/hash catalogue."""
import hashlib,json
from pathlib import Path,PurePosixPath
from urllib.request import urlopen
from urllib.error import HTTPError
BASE=json.loads(Path('configs/research-case.json').read_text())['production_url'].rstrip('/')
def fetch(path):
    with urlopen(BASE+'/'+path,timeout=30) as response:return response.read()
def restore():
    try:catalog=json.loads(fetch('publication-catalog.json'))
    except HTTPError as error:
        if error.code!=404:raise
        try:fetch('snapshot-manifest.json')
        except HTTPError as manifest_error:
            if manifest_error.code==404:
                print('First transition from legacy deployment; no previously pinned v2 cases.');return
            raise
        raise RuntimeError('Published v2 manifest exists without its case archive catalogue; refusing to drop pinned cases')
    if catalog.get('schema_version')!=1 or not isinstance(catalog.get('files'),dict):raise ValueError('Invalid publication catalogue')
    root=Path('build/restored/release');root.mkdir(parents=True,exist_ok=True)
    total=0
    for name,expected in catalog['files'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or len(path.parts)<3 or path.parts[0]!='snapshots':raise ValueError('Unsafe catalogue path')
        payload=fetch(name);total+=len(payload)
        if total>500_000_000:raise ValueError('Archive exceeds owner restore size limit')
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Published artifact hash mismatch: '+name)
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    pointer=Path('build/current')
    if pointer.exists():raise ValueError('Restore expects a clean CI workspace')
    pointer.symlink_to('restored/release',target_is_directory=True)
    print(f'Restored {len(catalog["files"])} pinned files with hashes verified')
if __name__=='__main__':restore()
