#!/usr/bin/env python3
"""
Shared-PCA HDBSCAN clustering: compare several systems in ONE PCA space.

Usage:
    python cluster_hdbscan.py [config.yaml]
"""

import os
import sys
import pickle
import contextlib
import warnings
import csv
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import mdtraj as md
import hdbscan
import matplotlib

matplotlib.use("Agg")  # Headless-safe for Slurm/HPC environments
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import load_config, parse_ndx, medoid_index, system_outdir, config_path


# ============================================================
# CONFIG & LOGGING
# ============================================================

def resolve_systems(cfg, config_file="config.yaml"):
    output_dir = config_path(config_file, cfg.get("output_dir", "."))
    if not cfg.get("systems"):
        raise ValueError("Add at least one entry under systems")
    
    systems = []
    mode = cfg.get("input_type", "trajectory")
    if mode not in ("trajectory", "csv"):
        raise ValueError("input_type must be 'trajectory' or 'csv'")
        
    for s in cfg["systems"]:
        base = config_path(config_file, s.get("dir", "."))
        entry = dict(
            name=s["name"],
            input_type=mode,
            outdir=system_outdir(output_dir, s),
            frame_interval=s.get("frame_interval", [0, None]),
            stride=s.get("stride", 1),
        )
        if mode == "trajectory":
            entry.update(
                topology=os.path.join(base, s["topology"]),
                index=os.path.join(base, s["index"]),
                trajectories=[os.path.join(base, t) for t in s["trajectories"]]
            )
        else:
            if not s.get("files"):
                raise ValueError(f"{s['name']}: files must contain at least one CSV")
            entry["files"] = [os.path.join(base, f) for f in s["files"]]
        systems.append(entry)
        
    names = [s["name"] for s in systems]
    outdirs = [os.path.abspath(s["outdir"]) for s in systems]
    
    if len(set(names)) != len(names):
        raise ValueError("System names must be unique")
    if len(set(outdirs)) != len(outdirs):
        raise ValueError("System output folders collide; give distinct names")
        
    return output_dir, systems


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def tee_stdout(logfile_path):
    log_f = open(logfile_path, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_f)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        log_f.close()


# ============================================================
# LOADING HELPERS
# ============================================================

def select_atoms(topfile, ndxfile, group_name, name_filter):
    top = md.load_topology(topfile)
    ndx_groups = parse_ndx(ndxfile)
    if group_name not in ndx_groups:
        raise KeyError(f"Group '{group_name}' not found in {ndxfile}.")
    
    base_idx = ndx_groups[group_name]
    if not base_idx or min(base_idx) < 0 or max(base_idx) >= top.n_atoms:
        raise ValueError(f"Index group '{group_name}' in {ndxfile} is invalid.")
        
    if name_filter.lower() != "all":
        query = " or ".join(f"name {n.strip()}" for n in name_filter.split("or"))
        filter_idx = top.select(query)
        atom_idx = np.intersect1d(base_idx, filter_idx)
    else:
        atom_idx = np.array(base_idx)
        
    atom_idx = np.sort(atom_idx).tolist()
    if not atom_idx:
        raise ValueError(f"No atoms selected from group '{group_name}'.")
    return atom_idx


def load_system(system, atom_idx):
    f0, f1 = system["frame_interval"]
    f0 = f0 or 0
    stride = system["stride"]
    
    segments, frame_mapping, traj_id = [], [], []
    for tfile in system["trajectories"]:
        with md.open(tfile) as f:
            total_file_frames = len(f)
        end = total_file_frames if f1 is None else min(f1, total_file_frames)
        start_idx = (f0 + stride - 1) // stride
        end_idx = (end + stride - 1) // stride
        
        if start_idx >= end_idx:
            warnings.warn(f"{system['name']}: {tfile} has no selected frames; skipping")
            continue

        tr_full = md.load(tfile, top=system["topology"], atom_indices=atom_idx, stride=stride)
        tr2 = tr_full[start_idx:end_idx]
        segments.append(tr2)
        traj_id.extend([tfile] * tr2.n_frames)
        for i in range(start_idx, start_idx + tr2.n_frames):
            frame_mapping.append((tfile, i * stride))

    if not segments:
        raise ValueError(f"{system['name']}: no frames selected from any trajectory")
        
    subtraj = segments[0].join(segments[1:]) if len(segments) > 1 else segments[0]
    return subtraj, frame_mapping, np.array(traj_id)


def load_csv_system(system, features):
    f0, f1 = system["frame_interval"]
    f0 = f0 or 0
    stride = system["stride"]
    
    rows, mapping = [], []
    for path in system["files"]:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for i, row in enumerate(reader):
                if i < f0 or (f1 is not None and i >= f1) or i % stride:
                    continue
                values = [float(row[col]) for col in features]
                rows.append(values)
                mapping.append((path, i))
                
    if not rows:
        raise ValueError(f"{system['name']}: no CSV rows selected")
    return np.asarray(rows), mapping


# ============================================================
# PIPELINE STEPS
# ============================================================

def load_all_systems(systems, ref_name, group_name, name_filter, features=None):
    print("=== Loading all systems ===")
    if ref_name not in {s["name"] for s in systems}:
        raise ValueError(f"Reference '{ref_name}' is not in systems")
    ordered = sorted(systems, key=lambda s: s["name"] != ref_name)

    system_data = {}
    reference_frame = None
    ref_n_atoms = None
    mode = systems[0]["input_type"]

    for system in ordered:
        name = system["name"]
        print(f"\n--- {name} ---")
        try:
            if mode == "trajectory":
                atom_idx = select_atoms(system["topology"], system["index"], group_name, name_filter)
                subtraj, frame_mapping, traj_id = load_system(system, atom_idx)
                X = subtraj.xyz.reshape(subtraj.n_frames, -1)
                print(f"  {subtraj.n_frames} frames, {len(atom_idx)} atoms")
            else:
                X, frame_mapping = load_csv_system(system, features)
                subtraj, traj_id = None, None
                print(f"  {len(X)} rows, {X.shape[1]} features")
        except Exception as exc:
            raise RuntimeError(
                f"System '{name}' cannot be loaded: {exc}"
            ) from exc

        if reference_frame is None:
            reference_frame = subtraj[0] if mode == "trajectory" else True
            ref_n_atoms = X.shape[1]
            if mode == "trajectory":
                print("  -> using this system's frame 0 as the SHARED reference")

        if mode == "trajectory":
            subtraj.superpose(reference_frame)
            X = subtraj.xyz.reshape(subtraj.n_frames, -1)
            
        system_data[name] = dict(system=system, subtraj=subtraj,
                                 frame_mapping=frame_mapping, traj_id=traj_id, X=X,
                                 features=features)

    return system_data


def fit_shared_pca(system_data, n_components_max, output_dir):
    print("\n=== Fitting shared PCA on pooled data ===")
    X_per_system = {name: d["X"] for name, d in system_data.items()}
    X_pooled = np.vstack(list(X_per_system.values()))
    print(f"  Pooled feature matrix: {X_pooled.shape}")

    n_components = min(n_components_max, *X_pooled.shape)
    pca = PCA(n_components=n_components)
    pca.fit(X_pooled)

    explained_pct = pca.explained_variance_ratio_ * 100
    cumulative_pct = np.cumsum(explained_pct)
    print(f"  Explained variance per PC: {np.round(explained_pct, 2)}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shared_pca_explained_variance_ratio.npy"), pca.explained_variance_ratio_)
    np.save(os.path.join(output_dir, "shared_pca_cumulative_variance_ratio.npy"), np.cumsum(pca.explained_variance_ratio_))
    with open(os.path.join(output_dir, "shared_pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)

    lines = ["PC\tExplained_%\tCumulative_%"]
    for i, (ev, cv) in enumerate(zip(explained_pct, cumulative_pct), start=1):
        lines.append(f"PC{i}\t{ev:.3f}\t{cv:.3f}")
    with open(os.path.join(output_dir, "shared_pca_variance_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return pca, X_per_system, explained_pct, cumulative_pct


def cluster_system(name, d, pca, X, explained_pct, cumulative_pct, min_cluster_fraction, medoid_chunk_size):
    system = d["system"]
    outdir = system["outdir"]
    os.makedirs(outdir, exist_ok=True)
    reps_dir = os.path.join(outdir, "reps")
    os.makedirs(reps_dir, exist_ok=True)

    with tee_stdout(os.path.join(outdir, "hdbscan_output.dat")):
        print(f"--- {name} ---")
        X_pca = pca.transform(X)
        n_frames = X_pca.shape[0]

        min_cluster_size = max(5, int(min_cluster_fraction * n_frames))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", core_dist_n_jobs=-1)
        labels = clusterer.fit_predict(X_pca)

        counts = Counter(labels)
        counts.pop(-1, None)
        relabel_map = {old: new for new, (old, _) in enumerate(counts.most_common())}
        labels = np.array([relabel_map.get(lab, -1) for lab in labels])
        probabilities = clusterer.probabilities_

        n_noise = int(np.sum(labels == -1))
        print(f"  Clusters={len(set(labels) - {-1})} | Noise={n_noise} ({100 * n_noise / n_frames:.1f}%)")

        np.save(os.path.join(outdir, "labels.npy"), labels)
        np.save(os.path.join(outdir, "pca_coords.npy"), X_pca)
        np.save(os.path.join(outdir, "probabilities.npy"), probabilities)

        # Clear representatives from an earlier run.
        for pattern in ("cluster_*_rep.pdb", "cluster_*_rep.csv"):
            for path in Path(reps_dir).glob(pattern):
                path.unlink()

        cluster_ids = sorted(lab for lab in set(labels) if lab != -1)
        frame_mapping = d["frame_mapping"]
        for lab in tqdm(cluster_ids, desc=f"  {name} reps", unit="cluster", file=sys.stdout):
            idxs = np.where(labels == lab)[0]
            med = medoid_index(X_pca, idxs, chunk_size=medoid_chunk_size)
            source, source_row = frame_mapping[med]
            
            if system["input_type"] == "trajectory":
                full_frame = md.load_frame(source, index=source_row, top=system["topology"])
                representative = os.path.join(reps_dir, f"cluster_{lab}_rep.pdb")
                full_frame.save_pdb(representative)
            else:
                representative = os.path.join(reps_dir, f"cluster_{lab}_rep.csv")
                with open(representative, "w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["source_file", "source_row"] + d["features"])
                    writer.writerow([source, source_row + 1] + X[med].tolist())

        _plot_cluster_scatter(name, outdir, X_pca, labels, cluster_ids, explained_pct)
    return X_pca, labels


def _plot_cluster_scatter(name, outdir, X_pca, labels, cluster_ids, explained_pct):
    plt.figure(figsize=(6, 6))
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], s=15, alpha=0.4, color="#BBBBBB", label="Noise")
        
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(cluster_ids):
        mask = labels == lab
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=25, color=cmap(i % 10), edgecolors="black", linewidth=0.4, label=f"Cluster {lab}")
        
    plt.xlabel(f"PC1 ({explained_pct[0]:.1f}%)")
    plt.ylabel(f"PC2 ({explained_pct[1]:.1f}%)")
    plt.title(name)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pca_scatter.png"), dpi=300)
    plt.close()


def plot_vs_reference(results, system_data, ref_name, explained_pct, names, overlay_prefix):
    print(f"\n=== Building per-system vs '{ref_name}' comparison plots ===")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 24, "axes.linewidth": 2, "legend.frameon": False})

    ref_X = results[ref_name]["X_pca"]
    all_xy = np.vstack([r["X_pca"][:, :2] for r in results.values()])
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()
    pad_x, pad_y = 0.05 * (x_max - x_min) or 1.0, 0.05 * (y_max - y_min) or 1.0
    xlim, ylim = (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)

    tab10 = plt.get_cmap("tab10")
    color_index = {name: i for i, name in enumerate(names)}
    
    for name, r in results.items():
        if name == ref_name:
            continue
        outdir = system_data[name]["system"]["outdir"]
        sys_X = r["X_pca"]

        fig, ax = plt.subplots(figsize=(8, 8), gridspec_kw=dict(left=0.18, right=0.95, top=0.90, bottom=0.14))
        ax.scatter(ref_X[:, 0], ref_X[:, 1], s=20, color="#BBBBBB", alpha=0.35, edgecolors="none", label=ref_name, zorder=2)
        ax.scatter(sys_X[:, 0], sys_X[:, 1], s=20, color=tab10(color_index[name] % 10), alpha=0.45, edgecolors="none", label=name, zorder=3)

        ax.set_xlabel(f"PC1 ({explained_pct[0]:.1f}%)", fontsize=26)
        ax.set_ylabel(f"PC2 ({explained_pct[1]:.1f}%)", fontsize=26)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{name} vs {ref_name}", fontsize=24, pad=14)
        ax.legend(loc="upper right", fontsize=14, frameon=True, facecolor="white", framealpha=0.9)

        out_path = os.path.join(outdir, overlay_prefix)
        fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out_path}.svg", bbox_inches="tight")
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Shared-PCA HDBSCAN clustering.")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir, systems = resolve_systems(cfg, args.config)
    
    group_name = cfg.get("index_group", "Protein")
    name_filter = cfg.get("atom_filter", "CA")
    pca_components = cfg.get("pca_components", 10)
    min_cluster_fraction = cfg.get("min_cluster_fraction", 0.001)
    medoid_chunk_size = cfg.get("medoid_chunk_size", 2000)
    ref_name = cfg.get("reference") or systems[0]["name"]

    system_data = load_all_systems(systems, ref_name, group_name, name_filter, cfg.get("features"))
    pca, X_per_system, explained_pct, cumulative_pct = fit_shared_pca(system_data, pca_components, output_dir)

    print("\n=== Per-system projection + clustering ===")
    results = {}
    for name, d in system_data.items():
        try:
            X_pca, labels = cluster_system(
                name, d, pca, X_per_system[name], explained_pct, cumulative_pct,
                min_cluster_fraction, medoid_chunk_size,
            )
            results[name] = dict(X_pca=X_pca, labels=labels)
        except Exception as exc:
            raise RuntimeError(
                f"Clustering failed for '{name}': {exc}. "
                "Outputs from this run may be incomplete."
            ) from exc

    plot_vs_reference(results, system_data, ref_name, explained_pct,
                      [s["name"] for s in systems],
                      cfg.get("overlay_prefix", "vs_reference_conformational_space"))

    print("\n=== DONE ===")
    print(f"Per-system outputs : {output_dir}/reps_<system>/")

if __name__ == "__main__":
    main()