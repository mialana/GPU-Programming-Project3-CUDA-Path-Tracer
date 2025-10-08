import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------
# Data collected from performance logs
# -------------------------------------

sizes = np.array([2**16, 2**20, 2**24, 2**28, 2**29])

# --- Prefix Scan (ms)
cpu_scan     = np.array([0.05, 0.91, 11.01, 129.8, 361.4])
thrust_scan  = np.array([0.09, 0.16, 0.50, 7.48, 11.28])
custom_scan  = np.array([0.38, 4.78, 1.13, 12.58, 24.72])

# --- Radix Sort (ms)
thrust_sort  = np.array([0.27, 0.31, 4.18, 29.93, 60.17])
custom_sort  = np.array([0.51, 0.72, 19.30, 314.37, 645.37])

# --- Stream Compaction (ms)
cpu_compact    = np.array([0.31, 4.01, 30.9, 558.3, 1037.5])
custom_compact = np.array([0.40, 5.21, 6.04, 31.77, 64.06])

# -------------------------------------
# Plot helper
# -------------------------------------
def plot_runtime(title, sizes, cpu=None, thrust=None, custom=None, ylabel="Time (ms)"):
    plt.figure(figsize=(7,5))
    if cpu is not None:
        plt.plot(sizes, cpu, 'o--', label="CPU", color="#9e9e9e")
    if thrust is not None:
        plt.plot(sizes, thrust, 's--', label="Thrust (baseline)", color="#64b5f6")
    if custom is not None:
        plt.plot(sizes, custom, 'd-', label="Custom (shared memory)", color="#ff7043")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Input size (elements)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3, which='both')
    plt.legend()
    plt.figtext(
        0.5, 0.01,
        "Measured on NVIDIA RTX 5070 Ti — CUDA block size: 128 threads",
        ha="center", fontsize=8, color="gray"
    )
    plt.tight_layout()

# -------------------------------------
# Generate three plots
# -------------------------------------
plot_runtime("Prefix Scan Performance", sizes, cpu_scan, thrust_scan, custom_scan)
plt.savefig("scan_performance.png", dpi=300)
plot_runtime("Radix Sort Performance", sizes, None, thrust_sort, custom_sort)
plt.savefig("sort_performance.png", dpi=300)
plot_runtime("Stream Compaction Performance", sizes, cpu_compact, None, custom_compact)
plt.savefig("compaction_performance.png", dpi=300)

plt.show()
