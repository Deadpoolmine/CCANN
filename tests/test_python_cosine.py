"""Cosine distance across native SSD and empty append-log indexes."""

import numpy as np
import pytest
import subprocess
import sys

import ccannpy


def test_cosine_graph_create_add_reload_and_merge(tmp_path):
    prefix = str(tmp_path / "graph")
    vectors = np.random.default_rng(23).normal(size=(300, 32)).astype(np.float32)
    vectors[0] = 0
    vectors[0, 0] = 10
    tags = np.arange(300, dtype=np.uint32)

    index = ccannpy.Index.create(prefix, vectors, tags, metric=ccannpy.Metric.COSINE, threads=2)
    assert isinstance(index._backend, ccannpy._NativeIndex)
    query = vectors[0].copy()
    ids, distances = index.search(query, 10)
    assert ids[0] == 0
    assert distances[0] == pytest.approx(0, abs=1e-5)
    for tag, distance in zip(ids, distances):
        expected = 1 - np.dot(query, vectors[tag]) / (np.linalg.norm(query) * np.linalg.norm(vectors[tag]))
        assert distance == pytest.approx(expected, abs=2e-4)

    added = np.zeros(32, dtype=np.float32)
    added[1] = -7
    index.add(added, 500)
    assert index.search(added * 3, 1)[0][0] == 500
    del index

    index = ccannpy.Index.load(prefix, metric=ccannpy.Metric.COSINE, threads=2)
    assert index.search(added, 1)[0][0] == 500
    with pytest.raises(ValueError, match="metric|metadata|incompatible"):
        ccannpy.Index.load(prefix, metric=ccannpy.Metric.L2)
    index.remove(0)
    merged = index.merge(str(tmp_path / "merged"))
    assert merged.search(added, 1)[0][0] == 500
    with pytest.raises(ValueError, match="metric|metadata|incompatible"):
        ccannpy.Index.load(str(tmp_path / "merged"), metric=ccannpy.Metric.L2)


def test_cosine_empty_index_persists_metric_and_exact_distances(tmp_path):
    prefix = str(tmp_path / "empty")
    index = ccannpy.Index.create(prefix, np.empty((0, 2), dtype=np.float32),
                                 np.empty(0, dtype=np.uint32), metric=ccannpy.Metric.COSINE)
    index.add(np.array([10, 0], dtype=np.float32), 1)
    index.add(np.array([0, 5], dtype=np.float32), 2)
    ids, distances = index.search(np.array([3, 0], dtype=np.float32), 2)
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_allclose(distances, [0, 1], atol=1e-6)
    del index

    restored = ccannpy.Index.load(prefix, metric=ccannpy.Metric.COSINE)
    np.testing.assert_array_equal(restored.search(np.array([3, 0], dtype=np.float32), 2)[0], [1, 2])
    with pytest.raises(ValueError, match="metric|metadata|incompatible"):
        ccannpy.Index.load(prefix, metric=ccannpy.Metric.L2)
    restored.remove(1)
    merged = restored.merge(str(tmp_path / "snapshot"))
    assert merged.search(np.array([0, 2], dtype=np.float32), 1)[0][0] == 2


def test_low_dimension_cosine_graph_survives_process_exit(tmp_path):
    prefix = str(tmp_path / "graph")
    vectors = np.random.default_rng(7).normal(size=(300, 4)).astype(np.float32)
    tags = np.arange(300, dtype=np.uint32)
    index = ccannpy.Index.create(prefix, vectors, tags, metric=ccannpy.Metric.COSINE, threads=2)
    del index

    script = """
import os, sys
import numpy as np
import ccannpy
index = ccannpy.Index.load(sys.argv[1], metric=ccannpy.Metric.COSINE, threads=2)
index.add(np.array([0, 0, 0, 5], dtype=np.float32), 500)
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", script, prefix], check=False, timeout=20)
    assert result.returncode == 17
    restored = ccannpy.Index.load(prefix, metric=ccannpy.Metric.COSINE, threads=2)
    assert restored.npoints == 301
    ids, distances = restored.search(np.array([0, 0, 0, 1], dtype=np.float32), 1)
    assert ids[0] == 500
    assert distances[0] == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize("empty", [False, True])
def test_cosine_rejects_zero_and_nonfinite_vectors_without_consuming_tag(tmp_path, empty):
    prefix = str(tmp_path / "index")
    dimension = 4 if empty else 32
    vectors = (np.empty((0, dimension), dtype=np.float32) if empty else
               np.random.default_rng(4).normal(size=(300, dimension)).astype(np.float32))
    tags = np.arange(len(vectors), dtype=np.uint32)
    if not empty:
        vectors[0] = 0
        with pytest.raises(ValueError, match="zero|finite|norm"):
            ccannpy.Index.create(prefix, vectors, tags, metric=ccannpy.Metric.COSINE)
        vectors[0, 0] = 1
    index = ccannpy.Index.create(prefix, vectors, tags, metric=ccannpy.Metric.COSINE, threads=2)
    zero = np.zeros(dimension, dtype=np.float32)
    with pytest.raises(ValueError, match="zero|finite|norm"):
        index.add(zero, 900)
    with pytest.raises(ValueError, match="zero|finite|norm"):
        index.search(zero, 1)
    with pytest.raises(ValueError, match="zero|finite|norm"):
        index.add(np.array([np.nan] + [0] * (dimension - 1), dtype=np.float32), 900)
    valid = np.zeros(dimension, dtype=np.float32)
    valid[0] = 1
    index.add(valid, 900)
    assert index.search(valid, 1)[0][0] in set(tags) | {900}
