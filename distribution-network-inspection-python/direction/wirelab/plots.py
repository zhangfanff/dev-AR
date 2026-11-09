import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for kind, color in [("gt", "#2171b5"), ("pred", "#d95f0e")]:
        for prefix, axis, xs in [
            ("bin_", axes[0], (10.0, 15.0, 20.0, 30.0)),
            ("extend_", axes[1], (1.0, 1.1, 1.2, 1.4)),
        ]:
            values = []
            for x in xs:
                v = (
                    "default"
                    if (prefix == "bin_" and x == 15)
                    or (prefix == "extend_" and x == 1.2)
                    else prefix + f"{x:g}"
                )
                match = [
                    r
                    for r in summary
                    if r["method"] == "original"
                    and r["mask_source"] == kind
                    and r["variant"] == v
                    and r["group"] == "all"
                ]
                values.append(
                    match[0]["direction_recall_at_5deg"]
                    if match and match[0]["direction_recall_at_5deg"] is not None
                    else np.nan
                )
            axis.plot(
                xs,
                values,
                "o-",
                color=color,
                label="Manual masks" if kind == "gt" else "Predicted masks",
            )
    axes[0].set_xlabel("Voting bin width (degrees)")
    axes[1].set_xlabel("Endpoint extension factor")
    for axis in axes:
        axis.set_ylabel("Direction recall within 5 degrees")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sensitivity.png", dpi=200)
    plt.close(fig)
    selected = [
        r
        for r in summary
        if r["method"] == "probabilistic"
        and r["variant"] == "default"
        and r["group"] == "all"
        and r["CI_coverage_all_references"] is not None
    ]
    if selected:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        labels = [r["mask_source"] for r in selected]
        axes[0].bar(
            labels, [r["CI_coverage_all_references"] for r in selected], color="#3182bd"
        )
        axes[0].axhline(0.95, color="black", ls="--", label="Nominal 95%")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Coverage over all reference directions")
        axes[0].legend(fontsize=8)
        widths = [
            r["CI_mean_width_deg"] if r["CI_mean_width_deg"] is not None else np.nan
            for r in selected
        ]
        axes[1].bar(labels, widths, color="#e6550d")
        axes[1].set_ylabel("Mean available interval width (degrees)")
        fig.tight_layout()
        fig.savefig(out / "uncertainty.png", dpi=200)
        plt.close(fig)
