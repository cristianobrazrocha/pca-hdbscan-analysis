#!/usr/bin/env python3
"""Small end-to-end smoke test; run with `python smoke_test.py`."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import mdtraj as md
import numpy as np
import yaml

from cluster_hdbscan import load_system, main as cluster_main, select_atoms
from plot_medoids import main as plot_main


def make_system(folder, offset, seed):
    folder.mkdir()
    top = md.Topology()
    chain = top.add_chain()
    for i in range(5):
        residue = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", md.element.carbon, residue)

    rng = np.random.default_rng(seed)
    xyz = np.zeros((40, 5, 3), dtype=np.float32)
    xyz[:, :, 0] = np.arange(5) * 0.38
    xyz[:, :, 1:] += rng.normal(scale=0.025, size=(40, 5, 2))
    xyz[:, 2, 1] += offset + np.repeat([0.0, 0.16], 20)
    trajectory = md.Trajectory(xyz, top)
    trajectory[0].save_gro(str(folder / "system.gro"))
    trajectory.save_xtc(str(folder / "traj.xtc"))
    (folder / "index.ndx").write_text("[ Protein ]\n1 2 3 4 5\n")


def run():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_system(root / "reference", 0.0, 10)
        make_system(root / "variant", 0.12, 11)
        systems = [
            dict(name=name, dir=str(root / folder), topology="system.gro",
                 index="index.ndx", trajectories=["traj.xtc"])
            for name, folder in [("reference", "reference"), ("variant", "variant")]
        ]
        config = dict(output_dir=str(root / "results"), reference="reference",
                      index_group="Protein", atom_filter="CA", pca_components=3,
                      min_cluster_fraction=0.15, cluster_ids=[0, 1], systems=systems)
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))

        selection = select_atoms(str(root / "reference/system.gro"),
                                 str(root / "reference/index.ndx"), "Protein", "CA")
        trimmed = dict(name="reference", topology=str(root / "reference/system.gro"),
                       trajectories=[str(root / "reference/traj.xtc")],
                       frame_interval=[3, 18], stride=4)
        trajectory, mapping, _ = load_system(trimmed, selection)
        assert trajectory.n_frames == 4
        assert [frame for _, frame in mapping] == [4, 8, 12, 16]
        for i, (_, frame) in enumerate(mapping):
            raw = md.load_frame(trimmed["trajectories"][0], frame, top=trimmed["topology"],
                                atom_indices=selection)
            np.testing.assert_allclose(trajectory[i].xyz, raw.xyz, atol=1e-5)

        # Mock sys.argv so argparse in the main scripts reads the test config path
        original_argv = sys.argv.copy()
        
        sys.argv = ["cluster_hdbscan.py", str(config_path)]
        cluster_main()
        
        sys.argv = ["plot_medoids.py", str(config_path)]
        plot_main()
        
        sys.argv = original_argv

        for system in systems:
            outdir = root / "results" / f"reps_{system['name']}"
            labels = np.load(outdir / "labels.npy")
            coords = np.load(outdir / "pca_coords.npy")
            assert labels.shape == (40,) and coords.shape == (40, 3)
            assert (outdir / "hdbscan_output.dat").exists()
        assert (root / "results/shared_pca_medoids.png").exists()
        print("Synthetic smoke test passed")


if __name__ == "__main__":
    run()