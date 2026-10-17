"""chart_500t.py (Die Lichtfänger 500T ColorChecker, 已知光谱 + 真实 Noritsu 扫描) 加消融开关 (ablation_flags)。"""
import sys
sys.path.insert(0, "/home/user/filmlab"); sys.path.insert(0, "/home/user/filmlab/tools/eval")
import ablation_flags as af; af.install()
import chart_500t
print("####", af.tag())
for sc in ("tungsten", "led5600"):
    chart_500t.main("/tmp/claude-0/-home-user-filmlab/f67b7686-eba5-58fc-b23b-cf537590be15/scratchpad/charts", sc)
