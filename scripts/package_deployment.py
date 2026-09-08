"""Package exactly the validated static release; no server-side recomputation."""
import json,shutil
from pathlib import Path
root=Path('build/current').resolve();dest=Path('build/deploy')
if dest.exists():shutil.rmtree(dest)
shutil.copytree(root,dest)
config=json.loads(Path('vercel.json').read_text())
for key in ['buildCommand','outputDirectory','installCommand']:config.pop(key,None)
config['buildCommand']='';config['installCommand']='';config['framework']=None
(dest/'vercel.json').write_text(json.dumps(config,indent=2)+'\n')
print('Prepared build/deploy from',root.name)
