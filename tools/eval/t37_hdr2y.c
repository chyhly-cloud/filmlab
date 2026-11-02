/* T37: Radiance .hdr (RGBE, 新旧两种行程编码) → 亮度 Y = 0.2126 R + 0.7152 G + 0.0722 B (float32, 与 rawio.LUMA 相同)。
   输出: int32 H, int32 W, 然后 H*W 个 float32 (行优先, 第 0 行为 +Y 方向 = 天顶)。
   T38: 第三个参数 rgb 时改为输出每像素 3 个 float32 (线性 R, G, B)。
   编译: gcc -O2 -o hdr2y t37_hdr2y.c -lm      用法: hdr2y in.hdr out.y [rgb] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int die(const char *m) { fprintf(stderr, "hdr2y: %s\n", m); exit(1); }

int main(int argc, char **argv) {
    if (argc != 3 && !(argc == 4 && !strcmp(argv[3], "rgb"))) die("usage: hdr2y in.hdr out.y [rgb]");
    int rgb = argc == 4;
    FILE *f = fopen(argv[1], "rb"); if (!f) die("open in");
    char line[4096]; int fmt_ok = 0;
    if (!fgets(line, sizeof line, f) || strncmp(line, "#?", 2)) die("not a radiance file");
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\r') break;
        if (!strncmp(line, "FORMAT=32-bit_rle_rgbe", 22)) fmt_ok = 1;
        if (!strncmp(line, "FORMAT=32-bit_rle_xyze", 22)) die("xyze not supported");
    }
    if (!fmt_ok) fprintf(stderr, "hdr2y: no FORMAT line, assuming rgbe\n");
    int H, W; char sy[4], sx[4];
    if (!fgets(line, sizeof line, f) || sscanf(line, "%2s %d %2s %d", sy, &H, sx, &W) != 4) die("bad resolution line");
    if (strcmp(sy, "-Y") || strcmp(sx, "+X")) die("only -Y H +X W orientation supported");
    if (H <= 0 || W <= 0 || H > 65536 || W > 131072) die("bad size");
    unsigned char *scan = malloc((size_t)W * 4), *buf = malloc((size_t)W);
    float *row = malloc((size_t)W * 3 * sizeof(float));
    FILE *o = fopen(argv[2], "wb"); if (!o) die("open out");
    int32_t hw[2] = {H, W}; fwrite(hw, 4, 2, o);
    for (int y = 0; y < H; y++) {
        int c0 = fgetc(f), c1 = fgetc(f), c2 = fgetc(f), c3 = fgetc(f);
        if (c3 == EOF) die("truncated");
        if (W >= 8 && W < 32768 + 1 && c0 == 2 && c1 == 2 && !(c2 & 0x80)) {
            if (((c2 << 8) | c3) != W) die("scanline width mismatch");
            for (int ch = 0; ch < 4; ch++) {
                int x = 0;
                while (x < W) {
                    int n = fgetc(f); if (n == EOF) die("truncated rle");
                    if (n > 128) {
                        n -= 128; int v = fgetc(f); if (v == EOF || x + n > W) die("bad run");
                        memset(buf + x, v, n); x += n;
                    } else {
                        if (n == 0 || x + n > W) die("bad literal");
                        if (fread(buf + x, 1, n, f) != (size_t)n) die("truncated literal");
                        x += n;
                    }
                }
                for (int i = 0; i < W; i++) scan[4 * i + ch] = buf[i];
            }
        } else {   /* 平铺 (无行程) */
            scan[0] = c0; scan[1] = c1; scan[2] = c2; scan[3] = c3;
            if (fread(scan + 4, 4, W - 1, f) != (size_t)(W - 1)) die("truncated flat");
        }
        for (int i = 0; i < W; i++) {
            int e = scan[4 * i + 3];
            double s = e == 0 ? 0.0 : ldexp(1.0, e - 136);
            if (rgb) {
                for (int c = 0; c < 3; c++) row[3 * i + c] = (float)(s * scan[4 * i + c]);
            } else {
                row[i] = (float)(s * (0.2126 * scan[4 * i] + 0.7152 * scan[4 * i + 1] + 0.0722 * scan[4 * i + 2]));
            }
        }
        fwrite(row, sizeof(float), rgb ? 3 * W : W, o);
    }
    fclose(o); fclose(f);
    return 0;
}
