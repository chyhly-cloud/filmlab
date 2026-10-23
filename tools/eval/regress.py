"""回归夹具: 用法 regress.py save|check 文件.npz — 多种配置渲染同一组图, 比较逐位一致。
环境变量: FILMLAB_DATA (含 rawcache/*.npy 的目录), LEGACY=1 (corrections=()), GENERATION, ONLY, NIMG, STEP。"""
import sys, os, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("REPO", "/home/user/filmlab"))
from filmlab.engine import Engine, Params
if os.environ.get("GENERATION"):
    from filmlab import generation; generation.use(os.environ["GENERATION"])
from filmlab.recipes import recipe
SP = os.environ.get("FILMLAB_DATA", "data_cache")
STEP = int(os.environ.get("STEP", "10"))
IMGS = ["blue_37569", "chi1", "n26405", "n29240", "indoor_44218", "day_12020"]
def imgs():
    for s in IMGS:
        f = glob.glob(f"{SP}/rawcache/{s}*.npy")
        if f: yield s, np.load(f[0])[::STEP, ::STEP] * (10 / 78) / 0.184
CONF_ALL = {
    "cs_frame": {**recipe("cinestill_800t"), "timing": "frame", "roll": 1, "frame": 3},
    "cs_std_neutral": {**recipe("cinestill_800t"), "timing": "standard", "scanner": "frontier_neutral", "roll": None},
    "cs_gray_noritsu": {**recipe("cinestill_800t"), "timing": "gray", "scanner": "noritsu", "roll": 2},
    "portra_frame": {**recipe("kodak_portra_400"), "timing": "frame", "roll": 1},
    "bw_frame": {**recipe("bw_pan"), "timing": "frame", "roll": 1},
    "portra_print": {**recipe("kodak_portra_400"), "output": "print", "roll": 1},
    "velvia": {**recipe("fujifilm_velvia_100"), "roll": 1},
}
CONF = {k: CONF_ALL[k] for k in os.environ["ONLY"].split(",")} if os.environ.get("ONLY") else CONF_ALL
if os.environ.get("NIMG"): IMGS = IMGS[:int(os.environ["NIMG"])]
out = {}
if os.environ.get("LEGACY"):
    CONF = {k: {**v, "corrections": ()} for k, v in CONF.items()}
for cn, c in CONF.items():
    for s, x in imgs():
        out[f"{cn}/{s}"] = Engine(Params(**c)).render(x)
if sys.argv[1] == "save":
    np.savez(sys.argv[2], **out); print("saved", len(out))
else:
    # 严格: 期望键必须全部渲染 (缺输入不能静默跳过), 逐位相同 (NaN 位置一致); 任何不符都以非零退出
    ref = np.load(sys.argv[2]); bad = 0
    want = set(ref.keys()) if not (os.environ.get("ONLY") or os.environ.get("NIMG")) \
        else {k for k in ref.keys() if k.split("/")[0] in CONF and k.split("/")[1] in IMGS}
    missing, extra = sorted(want - set(out)), sorted(set(out) - set(ref.keys()))
    for k in missing:
        print("MISSING", k)
    for k in extra:
        print("NO_REFERENCE", k)
    for k in sorted(set(out) & set(ref.keys())):
        v, r = out[k], ref[k]
        if v.shape != r.shape or not np.array_equal(v, r, equal_nan=True):
            bad += 1
            print("DIFF", k, np.nanmax(np.abs(v - r)) if v.shape == r.shape else f"shape {v.shape} vs {r.shape}")
    print("checked", len(set(out) & set(ref.keys())), "of", len(want), "differ", bad, "missing", len(missing))
    sys.exit(1 if (bad or missing or extra or not out) else 0)
