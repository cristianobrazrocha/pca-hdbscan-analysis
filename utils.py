"""Shared helpers for the shared-PCA conformational-comparison pipeline."""

from __future__ import annotations

import os
import re

import numpy as np
import yaml
from scipy.spatial.distance import cdist


def load_config(path):
    """Load the pipeline YAML config into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def config_path(path, value):
    """Resolve a user path relative to the config file, unless absolute."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(path)), value))


def slug(name):
    """Filesystem-safe version of a display name.

    'replica 1' -> 'replica_1', 'pH 7.4' -> 'ph_7_4'.
    """
    s = re.sub(r"[^0-9A-Za-z]+", "_", name.strip().lower())
    return s.strip("_") or "system"


def system_outdir(output_dir, system):
    """Where one system's results live. Both scripts use this so the
    clustering stage and the plotting stage always agree on the folder."""
    return os.path.join(output_dir, system.get("outdir", f"reps_{slug(system['name'])}"))


def parse_ndx(ndx_filename):
    """Parse a GROMACS .ndx file into ``{group_name: [0-based indices]}``.

    GROMACS index files are 1-based; indices are converted to 0-based so they
    can be used directly with mdtraj / numpy selections.
    """
    groups, current = {}, None
    with open(ndx_filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                current = line.strip("[] ").strip()
                groups[current] = []
            elif current is not None:
                groups[current].extend(int(x) - 1 for x in line.split())
    return groups


def medoid_index(coords, subset=None, chunk_size=2000):
    """Row index of the medoid of the selected points.

    The medoid minimises the summed Euclidean distance to every other selected
    point. Distances are computed in chunks so the full pairwise matrix is
    never materialised at once, keeping memory bounded for large clusters.

    Parameters
    ----------
    coords : (N, D) ndarray
    subset : 1-D int array, optional
        Rows of ``coords`` to consider. ``None`` uses every row.
    chunk_size : int

    Returns
    -------
    int
        An index into ``coords`` (not into ``subset``).
    """
    subset = np.arange(coords.shape[0]) if subset is None else np.asarray(subset)
    sub = coords[subset]
    n = sub.shape[0]
    if n == 0:
        raise ValueError("Cannot find a medoid of zero points")
    if chunk_size < 1:
        raise ValueError("medoid_chunk_size must be positive")
    # cdist returns float64: cap the working distance block near 64 MiB.
    chunk_size = min(chunk_size, max(1, (64 * 1024 * 1024) // (8 * n)))
    dist_sums = np.zeros(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dist_sums[start:end] = cdist(sub[start:end], sub, metric="euclidean").sum(axis=1)
    return int(subset[int(np.argmin(dist_sums))])
