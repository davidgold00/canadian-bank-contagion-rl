"""Immutable releases and one atomic pointer switch, never rollback over live files."""
from contextlib import contextmanager
from pathlib import Path
import fcntl, os, shutil, tempfile

@contextmanager
def release_transaction(pointer,snapshot_id):
    pointer=Path(pointer);pointer.parent.mkdir(parents=True,exist_ok=True)
    releases=pointer.parent/'releases';releases.mkdir(exist_ok=True)
    with (pointer.parent/'.publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if pointer.exists() and not pointer.is_symlink(): raise ValueError('Publication pointer must be a symlink; use build/current')
        stage=Path(tempfile.mkdtemp(prefix='.stage-',dir=releases))
        try:
            yield stage
            target=releases/snapshot_id
            if target.exists():
                from src.dashboard.case_site import tree_hashes
                if tree_hashes(stage)!=tree_hashes(target): raise ValueError('Release identity already exists with different contents')
                shutil.rmtree(stage)
            else: stage.rename(target)
            temp=pointer.with_name(pointer.name+'.next')
            if temp.is_symlink():temp.unlink()
            temp.symlink_to(os.path.relpath(target,pointer.parent),target_is_directory=True)
            os.replace(temp,pointer)
        finally:
            if stage.exists():shutil.rmtree(stage)
