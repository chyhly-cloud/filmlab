import sys, glob, os, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab")
from multiprocessing import Pool
from pathlib import Path
SP="/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"
out=Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
files=[f for f in sorted(glob.glob(SP+'/rawcache/*.npy')) if os.path.basename(f).startswith(tuple(os.environ.get('ONLY','').split(','))) ] if os.environ.get('ONLY') else [f for f in sorted(glob.glob(SP+"/rawcache/*.npy")) if not os.path.basename(f).startswith(("day_s7","day_x1","day_z6","day_ip","day_gr","day_q2"))]
def job(a):
    i,f=a
    from filmlab.engine import Engine, Params
    from filmlab.io import save_srgb
    from filmlab.recipes import recipe
    import json, zlib
    from filmlab.rawio import film_metering
    meta=json.load(open(SP+"/exifmeta.json")).get(Path(f).stem,[None,None])
    raw=np.load(f)[::2,::2]*(10/78)/0.184
    rgb,_=film_metering(raw,meta[0],rng=np.random.default_rng(zlib.crc32(Path(f).stem.encode())),t_original=meta[1])
    eng=Engine(Params(**{**recipe("cinestill_800t"),"timing":"frame","roll":1+i//36,"frame":i%36}))
    if os.environ.get("STRENGTH"): eng.scanner.strength=float(os.environ["STRENGTH"])
    save_srgb(eng.render(rgb),out/(Path(f).stem+".jpg")); return Path(f).stem
with Pool(int(sys.argv[2]) if len(sys.argv)>2 else 4) as p:
    for s in p.imap_unordered(job, list(enumerate(files))): pass
print("done", len(files))
