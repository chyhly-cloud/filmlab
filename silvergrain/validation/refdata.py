"""检验数据 (silvergrain/validation/data/): Schmidt 显微颗粒与曲线读数、T46–T53 的参考结果、真实扫描的记录值。只读。"""
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def csv_cols(name):
    """# 注释行之后的 CSV → {列: 数组} (数值列; 空格为 nan)。"""
    rows = [l for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    head = rows[0].split(",")
    out = {}
    for i, h in enumerate(head):
        vals = [r.split(",")[i] for r in rows[1:]]
        try:
            out[h] = np.array([float(v) if v else np.nan for v in vals])
        except ValueError:
            out[h] = vals
    return out


def csv_rows(name):
    return list(csv.DictReader(l for l in (DATA / name).read_text(encoding="utf-8").splitlines() if not l.startswith("#")))
