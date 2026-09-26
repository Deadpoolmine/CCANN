"""Platform checks for the native Python graph index."""

import sys

import numpy as np
import pytest

import ccannpy


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native wheel check")
def test_windows_wheel_exposes_native_index():
    assert ccannpy._NativeIndex.__module__ == "ccannpy._native"


def test_nonempty_index_uses_native_graph_and_pq(tmp_path):
    prefix = str(tmp_path / "index")
    vectors = np.random.default_rng(11).random((300, 64), dtype=np.float32)
    tags = np.arange(1000, 1300, dtype=np.uint32)

    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    assert isinstance(index._backend, ccannpy._NativeIndex)
    assert index.npoints == 300
    assert not (tmp_path / "index_ccann.flat").exists()
    for suffix in ("_disk.index", "_disk.index.tags", "_pq_compressed.bin", "_pq_pivots.bin"):
        assert (tmp_path / ("index" + suffix)).is_file()

    index.add(np.full(64, 3, dtype=np.float32), 2000)
    index.remove(1000)
    del index

    restored = ccannpy.Index.load(prefix, threads=2)
    assert isinstance(restored._backend, ccannpy._NativeIndex)
    assert restored.npoints == 300
    assert restored.search(np.full(64, 3, dtype=np.float32), 1)[0][0] == 2000


def test_native_graph_in_unicode_directory(tmp_path):
    directory = tmp_path / "\u4e2d\u6587\u7d22\u5f15"
    directory.mkdir()
    prefix = str(directory / "graph")
    vectors = np.random.default_rng(29).random((256, 64), dtype=np.float32)
    tags = np.arange(256, dtype=np.uint32)

    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    assert isinstance(index._backend, ccannpy._NativeIndex)
    assert index.search(vectors[0], 1)[0][0] == 0
    index.add(np.full(64, 3, dtype=np.float32), 300)
    index.remove(1)
    index.merge()
    del index

    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 256
    assert restored.search(np.full(64, 3, dtype=np.float32), 1)[0][0] == 300
