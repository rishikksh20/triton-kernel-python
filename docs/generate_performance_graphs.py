"""Regenerate the README's calculated H100 graphs; requires matplotlib.

Run: python docs/generate_performance_graphs.py
All values describe an idealized model, not measured GPU performance.
Reference: https://www.nvidia.com/en-gb/data-center/h100/
The listed 1,979 TFLOP/s FP16/BF16 sparse peak is twice the dense rate.
"""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


OUT = Path(__file__).resolve().parent / "images"
HBM_BYTES_PER_SECOND = 3.35e12
FP32_FLOPS_PER_SECOND = 67e12
DENSE_TENSOR_FLOPS_PER_SECOND = 1979e12 / 2
BLUE, GREEN, INK, MUTED = "#2563eb", "#047857", "#172b4d", "#475569"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.hashsalt": "triton-h100-readme-model",
    "svg.fonttype": "path",
})


def save_figure(fig, filename, title, description):
    """Save a standalone accessible SVG and a temporary inspection PNG."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, format="svg", facecolor="#f8fafc", metadata={"Date": None})
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    root = ET.parse(path).getroot()
    view_width, view_height = map(float, root.attrib["viewBox"].split()[2:])
    root.set("width", "1200")
    root.set("height", str(round(1200 * view_height / view_width)))
    root.set("role", "img")
    root.set("aria-labelledby", "title desc")
    title_el = ET.Element(f"{{{ns}}}title", {"id": "title"})
    title_el.text = title
    desc_el = ET.Element(f"{{{ns}}}desc", {"id": "desc"})
    desc_el.text = description
    root.insert(0, desc_el)
    root.insert(0, title_el)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)
    preview = Path(tempfile.gettempdir()) / path.with_suffix(".png").name
    fig.savefig(preview, dpi=110, facecolor="#f8fafc")
    plt.close(fig)
    print(f"Generated {path.name}; preview: {preview}")


def roofline():
    intensity = np.logspace(-2, 4, 900)
    memory_roof = HBM_BYTES_PER_SECOND * intensity / 1e12
    fp32_roof = np.minimum(FP32_FLOPS_PER_SECOND / 1e12, memory_roof)
    tensor_roof = np.minimum(DENSE_TENSOR_FLOPS_PER_SECOND / 1e12, memory_roof)
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.83, bottom=0.20)
    fig.suptitle("H100 SXM: arithmetic intensity changes the performance ceiling",
                 x=0.05, y=0.975, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.05, 0.923, "Calculated roofs and example bounds; these are not measured kernel results.",
             color=MUTED)
    ax.loglog(intensity, tensor_roof, color=GREEN, lw=3,
              label="FP16/BF16 Tensor Core dense model: 989.5 TFLOP/s")
    ax.loglog(intensity, fp32_roof, color=BLUE, lw=2.5, ls="--",
              label="Ordinary FP32 model: 67 TFLOP/s")
    for peak, color in ((FP32_FLOPS_PER_SECOND, BLUE),
                        (DENSE_TENSOR_FLOPS_PER_SECOND, GREEN)):
        ridge = peak / HBM_BYTES_PER_SECOND
        ax.vlines(ridge, 0.03, peak / 1e12, color=color, lw=1.3, ls=":")
        ax.scatter([ridge], [peak / 1e12], s=50, color=color, zorder=5)
    ax.text(20, 0.045, "FP32 ridge\n20 FLOP/byte", color=BLUE, ha="center", fontsize=11)
    ax.text(989.5 / 3.35, 0.045, "Tensor ridge\n~295 FLOP/byte", color=GREEN,
            ha="center", fontsize=11)

    add_i = 1 / 12
    add_p = HBM_BYTES_PER_SECOND * add_i / 1e12
    ax.scatter([add_i], [add_p], s=80, facecolors="white", edgecolors=BLUE,
               linewidths=2, zorder=6)
    ax.annotate("FP32 vector add\nI = 0.083; bound ~0.279 TFLOP/s",
                xy=(add_i, add_p), xytext=(0.018, 2.5), fontsize=11,
                arrowprops={"arrowstyle": "->", "color": MUTED})
    mat_i = 8192 / 4
    mat_p = DENSE_TENSOR_FLOPS_PER_SECOND / 1e12
    ax.scatter([mat_i], [mat_p], s=80, facecolors="white", edgecolors=GREEN,
               linewidths=2, zorder=6)
    ax.annotate("8192-square matmul\nIdeal-reuse I = 2048",
                xy=(mat_i, mat_p), xytext=(520, 150), fontsize=11,
                arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.set(xlim=(0.01, 10000), ylim=(0.03, 2500),
           xlabel="Arithmetic intensity at HBM (FLOP/byte; logarithmic scale)",
           ylabel="Model upper bound (TFLOP/s; logarithmic scale)")
    formatter = FuncFormatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, which="major", color="#cbd5e1", alpha=0.8)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.97)
    fig.text(0.05, 0.087, "Sloped region: bandwidth sets the roof.  Flat region: the chosen arithmetic path sets the roof.")
    fig.text(0.05, 0.041, "HBM: 3.35 TB/s. Dense Tensor estimate = 1,979 sparse TFLOP/s / 2. Real traffic and throughput vary.",
             fontsize=10, color=MUTED)
    save_figure(fig, "h100-roofline.svg", "Calculated H100 SXM rooflines",
                "Logarithmic model graph with a 3.35 TB/s HBM slope and distinct 67 TFLOP/s FP32 and "
                "989.5 TFLOP/s dense FP16/BF16 Tensor Core ceilings. Model example bounds place vector "
                "addition in the bandwidth region and an ideally reused large matmul in the compute region. "
                "Markers are calculated upper bounds, not benchmark observations.")


def time_bounds():
    n, size = 100_000_000, 8192
    cases = [
        ("FP32 vector add: 100 million elements", 12 * n, n,
         FP32_FLOPS_PER_SECOND, "HBM time dominates the streaming model"),
        ("FP16/BF16 matmul: 8192 x 8192 x 8192", 8 * size**2, 2 * size**3,
         DENSE_TENSOR_FLOPS_PER_SECOND, "Compute time dominates with ideal HBM reuse"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.3))
    fig.subplots_adjust(left=0.12, right=0.92, top=0.81, bottom=0.20, hspace=0.90)
    fig.suptitle("Two time estimates: which resource sets the lower bound?",
                 x=0.05, y=0.976, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.05, 0.922, "Calculated estimates, not measurements. The estimates can overlap; do not add the bars.",
             color=MUTED)
    for ax, (label, byte_count, flop_count, peak, conclusion) in zip(axes, cases):
        times = [byte_count / HBM_BYTES_PER_SECOND * 1e3, flop_count / peak * 1e3]
        ax.barh([1, 0], times, height=0.46, color=[BLUE, GREEN])
        for y, value in zip([1, 0], times):
            ax.text(value + 0.025, y, f"{value:.5f}" if value < 0.01 else f"{value:.3f}",
                    va="center", fontsize=12, color=INK)
        ax.set(yticks=[1, 0], yticklabels=["HBM time", "Compute time"], xlim=(0, 1.35),
               ylim=(-0.55, 1.55), xlabel="Time (milliseconds; shared linear scale)")
        ax.set_title(label + "\n" + conclusion, loc="left", fontsize=12, pad=12)
        ax.grid(axis="x", color="#cbd5e1", alpha=0.8)
        ax.set_axisbelow(True)
    fig.text(0.05, 0.107, "Model runtime lower bound = max(bytes / bandwidth, FLOPs / selected compute roof).")
    fig.text(0.05, 0.067, "FP32 uses the generous 67 TFLOP/s FMA peak; addition-only instruction throughput is lower.",
             fontsize=10, color=MUTED)
    fig.text(0.05, 0.027, "Matmul assumes FP16/BF16 inputs, FP32 output, and each input fetched once from HBM; overheads excluded.",
             fontsize=10, color=MUTED)
    save_figure(fig, "memory-compute-time-bounds.svg", "Memory versus compute time estimates",
                "Calculated H100 time lower bounds. A 100-million-element FP32 vector add has "
                "0.358 ms HBM time and a generous 0.00149 ms arithmetic bound using the published "
                "FP32 FMA peak; addition-only throughput is lower. An 8192-square FP16/BF16 matmul "
                "with FP32 output and ideal HBM input reuse has 0.160 ms memory time and 1.111 ms "
                "compute time. The larger estimate sets the model lower bound; real runtimes are not shown.")


if __name__ == "__main__":
    roofline()
    time_bounds()
