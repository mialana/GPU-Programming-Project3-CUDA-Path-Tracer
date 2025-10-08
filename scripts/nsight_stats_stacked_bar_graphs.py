import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Data from Nsight analysis
# ---------------------------------------------------------
bounces = np.arange(1, 9)

# kernel_scanIntraBlockShared
scan_compute = np.array([51.87, 52.14, 45.84, 45.78, 46.23, 34.85, 33.55, 38.18])
scan_memory  = np.array([76.12, 76.53, 67.15, 67.08, 67.75, 56.19, 49.05, 55.82])
scan_time    = np.array([52.58, 53.02, 29.95, 30.14, 30.34, 17.89, 20.13, 17.98])  # µs

# computeIntersections
intersect_compute = np.array([52.45, 46.74, 44.39, 44.01, 44.19, 43.12, 42.43, 42.18])
intersect_memory  = np.array([52.45, 46.74, 44.39, 44.01, 44.19, 43.12, 42.43, 42.18])
intersect_time    = np.array([22.23, 24.97, 18.76, 15.35, 15.00, 10.98, 9.46, 8.17])  # ms

# shadeFakeMaterial
shade_compute = np.array([12.38, 9.19, 7.80, 7.18, 7.88, 8.31, 8.41, 6.48])
shade_memory  = np.array([92.27, 54.13, 52.41, 68.82, 72.99, 74.15, 76.22, 63.84])
shade_time    = np.array([456.80, 643.62, 543.55, 68.70, 367.94, 295.23, 249.98, 39.46])  # µs

# Normalize time to ms for consistency
shade_time_ms = shade_time / 1000.0
scan_time_ms  = scan_time / 1000.0

# ---------------------------------------------------------
# Helper function for stacked plot
# ---------------------------------------------------------
def plot_stacked(intersect, shade, scan, ylabel, title, filename):
    width = 0.6
    plt.figure(figsize=(9,6))
    plt.bar(bounces, intersect, width, label="computeIntersections", color="#81c784")  # bottom
    plt.bar(bounces, shade, width, bottom=intersect, label="shadeFakeMaterial", color="#ffb74d")  # middle
    plt.bar(bounces, scan, width, bottom=intersect+shade, label="kernel_scanIntraBlockShared", color="#64b5f6")  # top
    plt.xlabel("Bounce Depth")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# Generate all three stacked charts
# ---------------------------------------------------------
plot_stacked(intersect_compute, shade_compute, scan_compute,
             "Compute (SM%)",
             "Kernel Comparison: SM Compute Utilization per Bounce",
             "stacked_compute.png")

plot_stacked(intersect_memory, shade_memory, scan_memory,
             "Memory (%)",
             "Kernel Comparison: Memory Utilization per Bounce",
             "stacked_memory.png")

plot_stacked(intersect_time, shade_time_ms, scan_time_ms,
             "Time (ms)",
             "Kernel Comparison: Duration per Bounce",
             "stacked_time.png")
