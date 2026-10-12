"""SG-MAE V2.0-A spatial graph regression and synthetic end-to-end tests."""

from __future__ import annotations

import json
import io
from contextlib import redirect_stderr
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from src.data.spatial_graph import (
    aligned_spatial_coordinates,
    build_frozen_shaml_graph,
    build_spatial_knn_graph,
    load_spatial_graph,
    save_spatial_graph,
)
from src.data.splitting import save_spatial_split, split_spatial_validation
from scripts.v2.build_graphs import main as graph_main


class SpatialGraphTests(unittest.TestCase):
    def test_known_neighbor_distances_and_radius(self):
        ids = ["c", "a", "b", "isolated"]
        xy = np.array([[0, 0], [3, 0], [0, 4], [200, 0]], dtype=float)
        graph = build_spatial_knn_graph(
            cell_ids=ids, coordinates_um=xy, node_indices=np.arange(4),
            sample_name="ShamR", partition="all", k=2, max_distance_um=5,
        )
        self.assertEqual(graph.num_nodes, 4)
        self.assertEqual(graph.num_edges, 6)
        self.assertEqual(graph.degree.tolist(), [2, 2, 2, 0])
        self.assertEqual(graph.neighbors[graph.indptr[0]:graph.indptr[1]].tolist(), [1, 2])
        np.testing.assert_allclose(graph.distances_um[:2], [3, 4])
        self.assertEqual(graph.summary()["num_isolated_nodes"], 1)

    def test_singleton_and_duplicate_coordinate_neighbors(self):
        single = build_spatial_knn_graph(
            cell_ids=["only"], coordinates_um=np.array([[0., 0.]]),
            node_indices=[0], sample_name="X", partition="all",
        )
        self.assertEqual(single.num_edges, 0)
        graph = build_spatial_knn_graph(
            cell_ids=["a", "b", "c"],
            coordinates_um=np.array([[0, 0], [0, 0], [0, 0]], dtype=float),
            node_indices=[0, 1, 2], sample_name="X", partition="all", k=1,
        )
        self.assertEqual(graph.num_edges, 3)
        self.assertTrue(np.all(graph.distances_um == 0))
        self.assertTrue(all(
            int(i) not in graph.neighbors[graph.indptr[i]:graph.indptr[i+1]]
            for i in range(graph.num_nodes)
        ))

    def test_tied_neighbors_use_stable_cell_id_order(self):
        ids = [f"cell_{i:02}" for i in range(19, -1, -1)]
        coords = np.zeros((len(ids), 2), dtype=np.float64)
        graph = build_spatial_knn_graph(
            cell_ids=ids, coordinates_um=coords,
            node_indices=np.arange(len(ids)), sample_name="ShamR",
            partition="all", k=3, max_distance_um=50,
        )
        for row, source in enumerate(ids):
            chosen = [graph.cell_ids[j] for j in graph.neighbors[
                graph.indptr[row]:graph.indptr[row + 1]
            ]]
            expected = sorted(cid for cid in ids if cid != source)[:3]
            self.assertEqual(chosen, expected)

    def test_exact_distance_tie_resolves_by_cell_id(self):
        ids = ["center", "z", "y", "x", "w"]
        coords = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        graph = build_spatial_knn_graph(
            cell_ids=ids, coordinates_um=coords,
            node_indices=list(range(5)), sample_name="ShamR",
            partition="all", k=2, max_distance_um=1,
        )
        near_center = [graph.cell_ids[j] for j in graph.neighbors[
            graph.indptr[0]:graph.indptr[1]
        ]]
        self.assertEqual(near_center, ["w", "x"])

    def test_cell_permutation_preserves_neighbor_cell_ids(self):
        ids = ["c0", "c1", "c2", "c3", "c4"]
        xy = np.array([[0,0], [1,0], [0,1], [1,1], [2,0]], dtype=float)
        order = np.array([3, 0, 4, 1, 2])
        first = build_spatial_knn_graph(
            cell_ids=ids, coordinates_um=xy, node_indices=np.arange(5),
            sample_name="test", partition="all", k=2, max_distance_um=5,
        )
        second = build_spatial_knn_graph(
            cell_ids=[ids[i] for i in order], coordinates_um=xy[order],
            node_indices=np.arange(5), sample_name="test", partition="all",
            k=2, max_distance_um=5,
        )
        def by_id(graph):
            return {
                cid: tuple(graph.cell_ids[j] for j in graph.neighbors[
                    graph.indptr[i]:graph.indptr[i+1]
                ])
                for i, cid in enumerate(graph.cell_ids)
            }
        self.assertEqual(by_id(first), by_id(second))

    def test_strict_byte_id_alignment_and_permuted_parquet(self):
        ids = ["a", "b", "c"]
        frame = pd.DataFrame({
            "cell_id": [b"c", b"a", b"b"],
            "x_centroid": [30., 10., 20.],
            "y_centroid": [3., 1., 2.],
        })
        _, coords = aligned_spatial_coordinates(ids, frame)
        np.testing.assert_allclose(coords, [[10., 1.], [20., 2.], [30., 3.]])
        _, rotated = aligned_spatial_coordinates(ids, frame.iloc[[1, 2, 0]])
        np.testing.assert_allclose(coords, rotated)
        with self.assertRaises(ValueError):
            aligned_spatial_coordinates(ids, frame.assign(cell_id=[b"a", b"a", b"c"]))
        with self.assertRaises(ValueError):
            aligned_spatial_coordinates(ids, frame.assign(cell_id=[b"a", b"b", b"missing"]))

    def test_invalid_inputs_and_split_provenance(self):
        ids = ["a", "b"]
        xy = np.array([[0.,0.], [1., 1.]])
        kwargs = dict(cell_ids=ids, coordinates_um=xy, sample_name="ShamL", partition="train", split_fingerprint="f")
        for selection in ([0, 0], [-1], [0, 2]):
            with self.assertRaises((ValueError, IndexError)):
                build_spatial_knn_graph(node_indices=selection, **kwargs)
        with self.assertRaises(ValueError):
            build_spatial_knn_graph(node_indices=[0,1], k=0, **kwargs)
        with self.assertRaises(ValueError):
            build_spatial_knn_graph(node_indices=[0,1], max_distance_um=float("nan"), **kwargs)
        with self.assertRaises(ValueError):
            build_spatial_knn_graph(node_indices=[0,1], coordinates_um=xy * float("nan"),
                                    cell_ids=ids, sample_name="X", partition="all")
        with self.assertRaises(ValueError):
            build_spatial_knn_graph(node_indices=[0], sample_name="ShamL",
                                    partition="train", cell_ids=ids, coordinates_um=xy)

    def test_exact_partition_separation_and_exclusion(self):
        ids = [f"cell_{i:02}" for i in range(20)]
        xy = np.array([[float((i // 5) * 100 + i % 5), float(i % 2)] for i in range(20)])
        table = pd.DataFrame({"cell_id": ids, "x_centroid": xy[:,0], "y_centroid": xy[:,1]})
        split = split_spatial_validation(
            ids, table, validation_fraction=0.25, block_size_um=100,
            buffer_um=10, seed=42, sample_name="ShamL",
        )
        train = build_frozen_shaml_graph(
            split=split, cell_ids=ids, coordinates_um=xy, partition="train",
            k=3, max_distance_um=50,
        )
        val = build_frozen_shaml_graph(
            split=split, cell_ids=ids, coordinates_um=xy, partition="validation",
            k=3, max_distance_um=50,
        )
        self.assertFalse(set(train.cell_ids) & set(val.cell_ids))
        self.assertTrue(set(train.cell_ids).isdisjoint(
            {ids[i] for i in split.excluded_indices}
        ))
        self.assertEqual(train.split_fingerprint, val.split_fingerprint)
        with self.assertRaises(ValueError):
            build_frozen_shaml_graph(
                split=split, cell_ids=ids, coordinates_um=xy + 0.5,
                partition="train",
            )
        with self.assertRaises(ValueError):
            build_frozen_shaml_graph(
                split=split, cell_ids=ids, coordinates_um=xy, partition="all",
            )

    def test_save_load_and_tamper_detection(self):
        graph = build_spatial_knn_graph(
            cell_ids=["z", "y", "x"],
            coordinates_um=np.array([[0,0],[1,0],[2,0]],dtype=float),
            node_indices=[2,0,1], sample_name="ShamR", partition="all", k=2,
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sample" / "all.npz"
            save_spatial_graph(graph, path)
            loaded = load_spatial_graph(path)
            self.assertEqual(graph.fingerprint, loaded.fingerprint)
            self.assertEqual(loaded.cell_ids, graph.cell_ids)
            np.testing.assert_array_equal(loaded.node_indices, [2, 0, 1])
            with self.assertRaises(FileExistsError):
                save_spatial_graph(graph, path)
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["num_directed_edges"] += 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_spatial_graph(path)

    def test_synthetic_cli_frozen_and_external(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            split_path = root / "frozen.json.gz"
            frames = {}

            def add_sample(prefix, offset):
                ids = [f"{prefix}_{i:02}" for i in range(24)]
                x = np.asarray([float((i//6)*120+(i%6)*3+offset) for i in range(24)])
                y = np.asarray([float(i%6) for i in range(24)])
                frame = pd.DataFrame({
                    "cell_id": [s.encode("utf8") for s in ids],
                    "x_centroid": x, "y_centroid": y,
                })
                sample_dir = data / f"{prefix}-mock"
                sample_dir.mkdir(parents=True)
                h5 = sample_dir / "cell_feature_matrix.h5"
                with h5py.File(h5, "w") as handle:
                    group = handle.create_group("matrix")
                    group.create_dataset("barcodes", data=np.asarray(ids, dtype="S32"))
                    group.create_dataset("shape", data=np.array([300, len(ids)]))
                (sample_dir / "cells.parquet").touch()
                frames[str(sample_dir / "cells.parquet")] = frame
                return ids, frame

            sham_ids, sham_frame = add_sample("ShamL", 0)
            add_sample("ShamR", 500)
            split = split_spatial_validation(
                sham_ids, sham_frame, validation_fraction=0.25,
                block_size_um=120, buffer_um=10, seed=3, sample_name="ShamL",
            )
            save_spatial_split(split, split_path)

            def synthetic_read(path, columns=None, **kwargs):
                return frames[str(path)].loc[:, list(columns)].copy()

            dest = root / "output"
            argv = [
                "--data-root", str(data), "--split", str(split_path),
                "--samples", "shaml", "shamr", "--output-dir", str(dest),
            ]
            with patch("pandas.read_parquet", side_effect=synthetic_read):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        graph_main(argv)
                result = graph_main(argv + ["--confirm-external"])
            self.assertTrue((dest / "graph_manifest.json").exists())
            self.assertFalse(any(p.name.startswith(".sgmae_v2_graph_stage_") for p in root.iterdir()))
            train = load_spatial_graph(dest / "shaml" / "train.npz")
            val = load_spatial_graph(dest / "shaml" / "validation.npz")
            ext = load_spatial_graph(dest / "shamr" / "all.npz")
            self.assertTrue(set(train.cell_ids).isdisjoint(val.cell_ids))
            self.assertEqual(set(train.node_indices.tolist()), set(split.train_indices.tolist()))
            self.assertEqual(set(val.node_indices.tolist()), set(split.validation_indices.tolist()))
            self.assertEqual(ext.num_nodes, 24)
            self.assertEqual(result["shaml_split_fingerprint"], split.fingerprint)
            with self.assertRaises(FileExistsError):
                graph_main(argv + ["--confirm-external"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
