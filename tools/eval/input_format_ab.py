"""输入格式对照: 同一场景、同一胶片扫描, RAW vs 相机内嵌 JPEG; 扫描仪条件 cur / s0 / s0_bare。
用法: exp.py 输出.json 场景ID,...   (场景 ID 见 epfl_800t_manifest.csv)"""
import sys, os, json, csv, numpy as np, colour, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/filmlab/tools/eval"); sys.path.insert(0, "/home/user/filmlab")
from multiprocessing import Pool
from PIL import Image
SP = "/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad"
EP = SP + "/epfl/data"
MAN = {r["scene_id"]: r for r in csv.DictReader(open("/home/user/filmlab/tools/eval/epfl_800t_manifest.csv"))}
EXTRA = {"cafe": (SP + "/pairs/sillystill/table_cafe_digital.ARW", SP + "/inputfmt/jpg/table_cafe_digital.jpg",
                  SP + "/pairs/sillystill/table_cafe_film.JPG", 1.5)}
COND = ["cur", "s05", "s0", "s0_bare"]

def job(a):
    sid, fmt = a
    import pair_compare as pc
    from filmlab import scanner as scm
    from filmlab.engine import Engine, Params
    from filmlab.recipes import recipe
    from filmlab.rawio import film_metering
    from filmlab.io import srgb_encode
    if sid in EXTRA:
        rawp, jpgp, scan, bias = EXTRA[sid]
    else:
        r = MAN[sid]; stem = os.path.basename(r["raw_file"])[:-4]
        rawp, jpgp, scan, bias = f"{EP}/{r['raw_file']}", f"{SP}/inputfmt/jpg/{stem}.jpg", f"{EP}/{r['film_file']}", 0.53
    rgb = pc.load_digital(rawp if fmt == "raw" else jpgp, 1400)[0]
    real = Image.open(scan).convert("RGB"); real.thumbnail((1600, 1600)); real = np.asarray(real, float) / 255
    import zlib; seed = zlib.crc32(sid.encode()) % 1000
    res = {}
    model = None
    for cond in COND:
        m, _ = film_metering(rgb, rng=np.random.default_rng(seed), bias_ev=bias)
        scm.SATURATION_MATRIX = SM0 if cond != "s0_bare" else np.eye(3)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": seed % 36}))
        if cond in ("s0", "s0_bare"): eng.scanner.strength = 0.0
        if cond == "s05": eng.scanner.strength = 0.5
        if cond == "s0_bare": eng.scanner.m = scm.SCANNERS["frontier_neutral"]
        sim = srgb_encode(np.clip(eng.render(m), 0, 1))
        try:
            w, mask, n = pc.align(sim, real)
        except Exception as e:
            return sid, fmt, {"error": str(e)}
        S, R = pc.compare(w, real, mask)
        de = colour.delta_E(S, R, method="CIE 2000")
        d = np.median(S - R, 0)
        res[cond] = dict(dE00=float(np.median(de)), dL=float(d[0]), da=float(d[1]), db=float(d[2]), n=int(len(S)), inl=n,
                         strength=float(eng.scanner.strength))
    return sid, fmt, res

SM0 = None
if __name__ == "__main__":
    from filmlab import scanner as _s
    SM0 = _s.SATURATION_MATRIX.copy()
    ids = sys.argv[2].split(",")
    with Pool(6) as p, open(sys.argv[1], "a") as fo:
        for sid, fmt, res in p.imap_unordered(job, [(i, f) for i in ids for f in ("raw", "jpg")]):
            fo.write(json.dumps([sid, fmt, res], ensure_ascii=False) + "\n"); fo.flush()
            print(sid, fmt, "  ".join(f"{c}: ΔE00 {v['dE00']:5.2f} Δb {v['db']:+5.1f}" for c, v in res.items() if c in COND) or res, flush=True)
