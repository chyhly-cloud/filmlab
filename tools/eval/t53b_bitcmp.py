"""T53b: t53b_bitwise.py 的两份输出 (HEAD / 改动后) 逐位比较 (docs/tasks/T53b_SILVER_BLOCK_BOUNDARY.md)。
不同时记: 改变的像素比例、8 位码值 (t52_measure.to_code, 取整) 改变的比例、码值差的最大 / 平均, 以及按 16 行分块的
块边界行 (行号 mod 16 = 0 或 15) 与块内行分开的平均。
用法: python3 -I t53b_bitcmp.py 仓库根 HEAD目录 新目录 输出.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[1]) / "tools" / "eval"))
import t52_measure as m52
A, B = Path(sys.argv[2]), Path(sys.argv[3])
res = {}
for fa in sorted(A.glob("*.npy")):
    fb = B / fa.name
    if not fb.exists():
        continue
    a, b = np.load(fa), np.load(fb)
    r = dict(bitwise_equal=bool(np.array_equal(a, b)))
    if not r["bitwise_equal"]:
        ca, cb = m52.to_code(a), m52.to_code(b)              # 8 位码值 (浮点, 取均值通道; 同 T52)
        d = cb - ca
        ch = np.any(a != b, -1)
        rows = np.arange(a.shape[0]) % 16
        edge = (rows == 0) | (rows == 15)
        r.update(frac_px_changed=float(ch.mean()),
                 frac_px_code8_changed=float((np.round(ca) != np.round(cb)).mean()),
                 max_abs_dcode=float(np.abs(d).max()), mean_dcode=float(d.mean()),
                 mean_dcode_block_edge_rows=float(d[edge].mean()), mean_dcode_interior_rows=float(d[~edge].mean()),
                 frac_changed_px_on_block_edge_rows=float(ch[edge].sum() / max(ch.sum(), 1)),
                 rel_mean_linear=float(b.mean() / a.mean() - 1))
    res[fa.stem] = r
neq = {k: v for k, v in res.items() if not v["bitwise_equal"]}
summary = dict(n=len(res), n_equal=len(res) - len(neq), n_changed=len(neq))
json.dump(dict(summary=summary, cases=res), open(sys.argv[4], "w"), indent=1)
print(summary)
for k, v in neq.items():
    print(f"{k:55s} px%={100*v['frac_px_changed']:6.2f} code8%={100*v['frac_px_code8_changed']:6.3f} "
          f"max|d|={v['max_abs_dcode']:.3f} mean={v['mean_dcode']:+.5f} edge={v['mean_dcode_block_edge_rows']:+.5f} "
          f"int={v['mean_dcode_interior_rows']:+.5f} onEdge={100*v['frac_changed_px_on_block_edge_rows']:.0f}%")
