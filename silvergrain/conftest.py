"""在本目录 (包的根) 运行 pytest 时, 让 `import silvergrain` 找到这个包本身 (扁平布局, 未安装时)。"""
import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
