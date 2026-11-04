#include <stdint.h>
#include <math.h>

/* Evaluate continuous marked disks at cell centers on a periodic square. */
void silver_counts(uint16_t *low, uint16_t *high, int n, double pitch,
                   const double *particles, int count) {
    for (int p = 0; p < count; ++p) {
        const double *grain = particles + p * 4;
        double x = grain[0] / pitch - 0.5;
        double y = grain[1] / pitch - 0.5;
        double radius = grain[2] / pitch;
        for (int row = (int)ceil(y - radius); row <= (int)floor(y + radius); ++row) {
            double dy = row - y;
            double width = sqrt(fmax(0, radius * radius - dy * dy));
            int first = (int)ceil(x - width);
            int last = (int)floor(x + width);
            int offset = ((row + n) % n) * n;
            for (int col = first; col <= last; ++col) {
                int i = offset + (col + n) % n;
                high[i]++;
                if (grain[3] < 0.682 / 1.576)
                    low[i]++;
            }
        }
    }
}
