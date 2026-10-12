#!/usr/bin/env python3
"""
Summary comparison figure across systems.

Usage:
    python plot_medoids.py [config.yaml]
"""

import os
import sys
import warnings
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import load_config, medoid_index, system_outdir, config_path

# ---- style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 28,
    "axes.linewidth": 2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
})

SUBPLOT_SIZE = 8
TABLE_SCALE = 1.3


def boost_saturation(rgb, factor=1.25):
    hsv = mcolors.rgb_to_hsv(rgb[:3])
    hsv[1] = min(hsv[1] * factor, 1.0)
    return mcolors.hsv_to_rgb(hsv)


def stats_from_labels(labels, cluster_ids):
    total = len(labels)
    cluster_labels = set(labels) - {-1}
    return dict(n_clusters=len(cluster_labels),
                pct_clustered=100.0 * np.count_nonzero(labels != -1) / total,
                cluster_pcts={cid: 100.0 * np.count_nonzero(labels == cid) / total
                              if cid in cluster_labels else np.nan
                              for cid in cluster_ids})


def add_summary_table(ax, names, stats, cluster_ids, colors):
    ax.axis("off")
    columns = ["System", "C", "%C"] + [f"%C{cid}" for cid in cluster_ids]

    rows = []
    for name in names:
        st = stats[name]
        if st is None:
            rows.append([name, "N/A", "N/A"] + ["N/A"] * len(cluster_ids))
        else:
            rows.append(
                [name, f"{st['n_clusters']}", f"{st['pct_clustered']:.1f}%"]
                + [(f"{st['cluster_pcts'][c]:.1f}%" if not np.isnan(st['cluster_pcts'][c]) else "N/A")
                   for c in cluster_ids]
            )

    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(20 * TABLE_SCALE)
    table.scale(1.0, 1.8 * TABLE_SCALE)
    n_cols = len(columns)
    
    for c in range(n_cols):
        cell = table[0, c]
        cell.set_text_props(weight="bold", fontsize=22 * TABLE_SCALE)
        cell.set_facecolor("#e0e0e0")
    for r, name in enumerate(names, start=1):
        table[r, 0].set_text_props(weight="bold", color=colors[name])


def main():
    parser = argparse.ArgumentParser(description="Plot medoids from shared-PCA.")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = config_path(args.config, cfg.get("output_dir", "."))
    systems = cfg["systems"]
    names = [s["name"] for s in systems]
    folders = {s["name"]: system_outdir(output_dir, s) for s in systems}
    cluster_ids = cfg.get("cluster_ids", [0, 1])
    out_prefix = cfg.get("output_prefix", "shared_pca_medoids")

    cmap = plt.get_cmap("tab10")
    colors = {name: boost_saturation(cmap(i % 10)) for i, name in enumerate(names)}

    ev_path = os.path.join(output_dir, "shared_pca_explained_variance_ratio.npy")
    if os.path.exists(ev_path):
        ev = np.load(ev_path) * 100
        pc1_label, pc2_label = (f"PC1 ({ev[0]:.1f}%)", f"PC2 ({ev[1]:.1f}%)") if len(ev) >= 2 else ("PC1", "PC2")
    else:
        pc1_label, pc2_label = "PC1", "PC2"

    coords, labels, stats = {}, {}, {}
    for name in names:
        folder = folders[name]
        try:
            xy = np.load(os.path.join(folder, "pca_coords.npy"), allow_pickle=False)
            labs = np.load(os.path.join(folder, "labels.npy"), allow_pickle=False)
        except OSError as exc:
            warnings.warn(f"Skipping '{name}': {exc}")
            continue
        coords[name], labels[name] = xy, labs
        stats[name] = stats_from_labels(labs, cluster_ids)
        
    names = [name for name in names if name in coords]

    all_xy = np.vstack([xy[:, :2] for xy in coords.values()])
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()
    px, py = 0.05 * (x_max - x_min) or 1.0, 0.05 * (y_max - y_min) or 1.0
    xlim, ylim = (x_min - px, x_max + px), (y_min - py, y_max + py)

    ncols, n_sys = len(cluster_ids), len(names)
    table_h = 0.6 * SUBPLOT_SIZE * TABLE_SCALE * max(1.0, n_sys / 7.0)
    fig_h = SUBPLOT_SIZE + table_h
    fig = plt.figure(figsize=(SUBPLOT_SIZE * ncols, fig_h))
    gs = fig.add_gridspec(2, ncols, height_ratios=[table_h / fig_h, 1 - table_h / fig_h],
                          left=0.08, right=0.97, top=0.94, bottom=0.12, hspace=0.3, wspace=0.28)

    add_summary_table(fig.add_subplot(gs[0, :]), names, stats, cluster_ids, colors)

    for i, cid in enumerate(cluster_ids):
        ax = fig.add_subplot(gs[1, i])
        for name in names:
            mask = labels[name] == cid
            if not np.any(mask):
                continue
            pts = coords[name][mask]
            med = pts[medoid_index(pts), :2]
            ax.scatter(med[0], med[1], s=260, facecolors=colors[name], edgecolors="black", linewidth=1.2, zorder=3)
            
        ax.set_xlabel(pc1_label, fontsize=32)
        ax.set_ylabel(pc2_label, fontsize=32)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.5, 0.96, f"Cluster {cid}", transform=ax.transAxes, fontsize=28, fontweight="bold", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.75), zorder=5)

    for ext in ["png", "svg"]:
        fig.savefig(os.path.join(output_dir, f"{out_prefix}.{ext}"), bbox_inches="tight", dpi=300 if ext == "png" else "figure")
    plt.close(fig)
    print(f"Saved {out_prefix}.png / .svg in {output_dir}")

if __name__ == "__main__":
    main()