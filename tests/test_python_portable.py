from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import numpy as np
import pytest

import ccannpy


@pytest.fixture
def portable_backend(monkeypatch):
    monkeypatch.setattr(ccannpy, "_NativeIndex", None)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows wheel import check")
def test_windows_wheel_uses_portable_backend():
    assert ccannpy._NativeIndex is None


def test_portable_index_round_trip_and_snapshot(portable_backend, tmp_path):
    prefix = str(tmp_path / "index")
    vectors = np.eye(300, 8, dtype=np.float32)
    tags = np.arange(1000, 1300, dtype=np.uint32)
    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    assert index.npoints == 300
    assert index.dimension == 8

    inserts = [(np.full(8, float(tag), dtype=np.float32), tag) for tag in range(2000, 2010)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: index.add(*item), inserts))
        results = list(pool.map(lambda item: index.search(item[0], 1)[0][0], inserts))
    assert results == list(range(2000, 2010))

    index.remove(1000)
    index.remove(2000)
    assert index.merge() is index
    snapshot_prefix = str(tmp_path / "snapshot")
    snapshot = index.merge(snapshot_prefix)
    assert snapshot.npoints == index.npoints == 308
    with pytest.raises(ValueError, match="already exists"):
        index.merge(snapshot_prefix)
    del index

    loaded = ccannpy.Index.load(prefix, threads=2)
    assert loaded.npoints == 308
    assert 1000 not in loaded.search(vectors[0], 10)[0]
    assert loaded.search(inserts[1][0], 1)[0][0] == 2001
    loaded.add(np.full(8, 50, dtype=np.float32), 3000)
    assert snapshot.npoints == 308
    assert ccannpy.Index.load(snapshot_prefix).npoints == 308


def test_portable_index_rejects_invalid_input(portable_backend, tmp_path):
    prefix = str(tmp_path / "index")
    vectors = np.zeros((300, 4), dtype=np.float32)
    tags = np.arange(300, dtype=np.uint32)
    with pytest.raises(ValueError):
        ccannpy.Index.create(prefix, vectors, tags, metric=ccannpy.Metric.COSINE)
    index = ccannpy.Index.create(prefix, vectors, tags)
    with pytest.raises(ValueError, match="Tag already exists"):
        index.add(vectors[0], 0)
    with pytest.raises(ValueError):
        index.search(np.zeros(3, dtype=np.float32), 1)
    with pytest.raises(ValueError):
        ccannpy.Index.load(prefix, metric=ccannpy.Metric.COSINE)


def test_portable_index_recovers_after_unclean_exit(portable_backend, tmp_path):
    prefix = str(tmp_path / "index")
    vectors = np.zeros((300, 4), dtype=np.float32)
    tags = np.arange(300, dtype=np.uint32)
    index = ccannpy.Index.create(prefix, vectors, tags)
    del index
    script = """
import os, sys
import numpy as np
import ccannpy
index = ccannpy.Index.load(sys.argv[1])
index.add(np.array([1, 2, 3, 4], dtype=np.float32), 9001)
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, prefix], check=True, timeout=15)
    restored = ccannpy.Index.load(prefix)
    assert restored.npoints == 301
    assert restored.search(np.array([1, 2, 3, 4], dtype=np.float32), 1)[0][0] == 9001
