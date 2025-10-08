import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# -------------------------------------
# Unterminated ray counts (depth 1–8)
# -------------------------------------
scenes = ["JelloShelf", "TentaclePlant", "Dragon", "YugiohClockArc"]

rays_open = np.array([
    [640000, 194769, 55075, 36754, 36599, 12470, 6974, 5160, 1820],   # Jello open
    [640000, 179413, 33394, 15300, 15243, 3953, 2084, 1544, 646],     # Tentacle open
    [640000, 192776, 47129, 26569, 26354, 8025, 4173, 2802, 1270],    # Dragon open
    [640000, 195140, 29928, 12252, 12162, 2597, 1598, 1123, 329],     # Yugioh open
])

rays_closed = np.array([
    [640000, 627655, 449847, 363658, 358471, 254970, 217900, 186876, 23226],   # Jello closed
    [640000, 627655, 459953, 363553, 358258, 248666, 210695, 179865, 23259],   # Tentacle closed
    [640000, 627655, 457972, 358502, 353374, 244627, 208654, 179845, 21879],   # Dragon closed
    [640000, 627655, 464095, 371712, 366407, 259462, 221009, 189319, 23438],   # Yugioh closed
])

triangles = np.array([1096, 5954, 19332, 51200])
bounces = np.arange(0, 9)

# -------------------------------------
# Plot setup
# -------------------------------------
fig, ax = plt.subplots(figsize=(8,6))
colors = ["#2196f3", "#4caf50", "#ab47bc", "#ff7043"]

for i, scene in enumerate(scenes):
    ax.plot(bounces, rays_open[i], '--', color=colors[i],
            label=f"{scene} (open, {triangles[i]} tris)")
    ax.plot(bounces, rays_closed[i], '-', color=colors[i],
            label=f"{scene} (closed)")

ax.set_xlabel("Bounce depth")
ax.set_ylabel("Unterminated rays")
ax.set_title("Stream Compaction: Unterminated Rays per Bounce")
ax.set_xticks(bounces)

# Adjust scale: logarithmic but with visible separation at high end
ax.set_yscale("symlog")
ax.set_ylim([500, 1000000])

formatter = ScalarFormatter(useMathText=False)
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

ax.grid(alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig("stream_compaction_absolute.png", dpi=300, bbox_inches="tight")
plt.show()