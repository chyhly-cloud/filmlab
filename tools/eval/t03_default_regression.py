"""Compare synthetic outputs with pre-T03 82c3701 hashes (same numeric runtime)."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from filmlab import generation
from filmlab.engine import Engine,Params
from filmlab.recipes import recipe
configs={
 'cs_frame':{**recipe('cinestill_800t'),'timing':'frame','roll':1,'frame':3},
 'cs_standard':{**recipe('cinestill_800t'),'timing':'standard','roll':None},
 'cs_gray':{**recipe('cinestill_800t'),'timing':'gray','scanner':'noritsu','roll':2},
 'portra_frame':{**recipe('kodak_portra_400'),'timing':'frame','roll':1},
 'bw_frame':{**recipe('bw_pan'),'timing':'frame','roll':1},
 'portra_print':{**recipe('kodak_portra_400'),'output':'print','roll':1},
 'velvia':{**recipe('fujifilm_velvia_100'),'roll':1}}
inputs=[np.random.default_rng(i).uniform(.001,2,(8,9,3)) for i in range(6)]
out={}
for g in ('current','next','legacy'):
 generation.use(g)
 for c,kw in configs.items():
  for i,x in enumerate(inputs):out[f'{g}/{c}/{i}']=Engine(Params(**kw)).render(x)
import json,hashlib
expected=json.loads((ROOT/'tests/t03_default_hashes.json').read_text())
assert set(expected)==set(out)
for k,v in out.items():
 assert hashlib.sha256(v.tobytes()).hexdigest()==expected[k], k
print('PASS baseline SHA256 bitwise synthetic regression: 126/126')
