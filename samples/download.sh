#!/bin/sh
# 测试用 RAW 样张 (raw.pixls.us, CC0)
cd "$(dirname "$0")"
curl -sSL -o a.NEF "https://raw.pixls.us/data/Nikon/Z%206/DSC_0750.NEF"
curl -sSL -o b.RAF "https://raw.pixls.us/data/Fujifilm/X-T3/AFXT2720.RAF"
curl -sSL -o c.RW2 "https://raw.pixls.us/data/Panasonic/DC-S1/P1033119.RW2"
curl -sSL -o d.CR3 "https://raw.pixls.us/data/Canon/EOS%20R/Canon_EOS_R_CRAW_ISO_100.CR3"
