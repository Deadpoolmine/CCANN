import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

import ccannpy


@pytest.fixture
def ssd_index(tmp_path):
    rng = np.random.default_rng(17)
    vectors = rng.random((300, 64), dtype=np.float32)
    tags = np.arange(1000, 1300, dtype=np.uint32)
    prefix = str(tmp_path / "index")
    return prefix, vectors, tags


def test_ssd_python_round_trip(tmp_path):
    rng = np.random.default_rng(7)
    vectors = rng.random((300, 64), dtype=np.float32)
    tags = np.arange(1000, 1300, dtype=np.uint32)
    prefix = str(tmp_path / "index")

    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    assert index.dimension == 64
    assert index.npoints == 300
    ids, _ = index.search(vectors[0], 5)
    assert 1000 in ids

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda tag: index.add(vectors[0].copy(), tag), range(2000, 2008)))
    assert index.npoints == 308
    index.remove(2000)
    assert index.npoints == 307
    with pytest.raises(ValueError):
        index.add(vectors[0], 1000)
    with pytest.raises(ValueError):
        index.add(np.zeros(32, dtype=np.float32), 2001)
    index.save()
    del index

    loaded = ccannpy.Index.load(prefix, threads=2)
    assert loaded.npoints == 307
    assert loaded.dimension == 64
    ids, _ = loaded.search(vectors[0], 5)
    assert 1000 in ids
    assert 2000 not in ids
    with pytest.raises(ValueError):
        ccannpy.Index.load(prefix, metric=ccannpy.Metric.COSINE)
    del loaded

    with open(prefix + "_pq_compressed.bin", "r+b") as file:
        file.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="Corrupt"):
        ccannpy.Index.load(prefix)


def test_multithreaded_insert_and_search_survive_restart(ssd_index):
    prefix, vectors, tags = ssd_index
    index = ccannpy.Index.create(prefix, vectors, tags, threads=4)
    new_vectors = np.eye(12, 64, dtype=np.float32) * 5
    new_tags = range(2000, 2012)
    valid_tags = set(range(1000, 1300)) | set(new_tags)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: index.add(*item), zip(new_vectors, new_tags)))

    def search_one(active_index, vector):
        ids, distances = active_index.search(vector, 5)
        assert ids.shape == distances.shape == (5,)
        assert set(ids).issubset(valid_tags)
        assert np.all(np.isfinite(distances))
        assert np.all(np.diff(distances) >= -1e-5)
        return ids

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda vector: search_one(index, vector), new_vectors))
        base_results = list(pool.map(lambda vector: search_one(index, vector), vectors[:12]))
    assert sum(1000 + offset in ids for offset, ids in enumerate(base_results)) >= 8

    assert index.npoints == 312
    index.save()
    del index
    persisted_tags = np.fromfile(prefix + "_disk.index.tags", dtype=np.uint32)[2:]
    assert set(new_tags).issubset(set(persisted_tags))
    restored = ccannpy.Index.load(prefix, threads=4)
    assert restored.npoints == 312
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda vector: search_one(restored, vector), new_vectors))


def test_multithreaded_search_while_inserting(ssd_index):
    prefix, vectors, tags = ssd_index
    index = ccannpy.Index.create(prefix, vectors, tags, threads=4)
    barrier = Barrier(4)
    new_vectors = np.eye(8, 64, dtype=np.float32) * 5

    def insert_batch(start):
        barrier.wait()
        for offset in range(start, start + 4):
            index.add(new_vectors[offset], 3000 + offset)

    def search_batch(start):
        barrier.wait()
        for offset in range(start, start + 8):
            query_id = offset % 8
            ids, distances = index.search(vectors[query_id], 5)
            assert ids.shape == distances.shape == (5,)
            assert np.all(np.isfinite(distances))
            assert np.all(np.diff(distances) >= -1e-5)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(insert_batch, 0), pool.submit(insert_batch, 4),
                   pool.submit(search_batch, 0), pool.submit(search_batch, 8)]
        for future in futures:
            future.result(timeout=15)

    assert index.npoints == 308
    index.save()
    del index
    persisted_tags = np.fromfile(prefix + "_disk.index.tags", dtype=np.uint32)[2:]
    assert set(range(3000, 3008)).issubset(set(persisted_tags))
    restored = ccannpy.Index.load(prefix, threads=4)
    assert restored.npoints == 308
    for vector in new_vectors:
        ids, _ = restored.search(vector, 5)
        assert ids.shape == (5,)
        assert set(ids).issubset(set(range(1000, 1300)) | set(range(3000, 3008)))


def test_insert_from_empty_and_reload(tmp_path):
    prefix = str(tmp_path / "empty")
    index = ccannpy.Index.create(prefix, np.empty((0, 4), dtype=np.float32),
                                 np.empty(0, dtype=np.uint32), threads=2)
    assert index.npoints == 0
    ids, distances = index.search(np.zeros(4, dtype=np.float32), 1)
    assert len(ids) == len(distances) == 0

    vectors = np.eye(4, dtype=np.float32)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: index.add(*item), zip(vectors, range(10, 14))))
    assert index.npoints == 4
    ids, distances = index.search(vectors[2], 2)
    assert ids[0] == 12
    assert distances[0] == 0
    index.remove(11)
    index.save()
    del index

    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 3
    ids, _ = restored.search(vectors[1], 3)
    assert 11 not in ids
    assert set(ids) == {10, 12, 13}
    restored.add(vectors[1], 14)
    assert restored.search(vectors[1], 1)[0][0] == 14


@pytest.mark.parametrize("empty", [False, True])
def test_recover_after_process_kill(tmp_path, empty):
    prefix = str(tmp_path / "index")
    vectors = (np.empty((0, 64), dtype=np.float32) if empty else
               np.random.default_rng(19).random((300, 64), dtype=np.float32))
    tags = np.empty(0, dtype=np.uint32) if empty else np.arange(1000, 1300, dtype=np.uint32)
    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    del index

    script = """
import os, signal, sys
import numpy as np
import ccannpy
index = ccannpy.Index.load(sys.argv[1], threads=2)
index.add(np.full(64, 7, dtype=np.float32), 9001)
os.kill(os.getpid(), signal.SIGKILL)
"""
    result = subprocess.run([sys.executable, "-c", script, prefix], check=False, timeout=20)
    assert result.returncode == -signal.SIGKILL
    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == len(vectors) + 1
    ids, distances = restored.search(np.full(64, 7, dtype=np.float32), 1)
    assert ids[0] == 9001
    assert distances[0] == 0


def test_empty_index_truncates_incomplete_record(tmp_path):
    prefix = str(tmp_path / "index")
    index = ccannpy.Index.create(prefix, np.empty((0, 4), dtype=np.float32),
                                 np.empty(0, dtype=np.uint32))
    vector = np.array([1, 2, 3, 4], dtype=np.float32)
    index.add(vector, 41)
    del index

    with open(prefix + "_ccann.flat", "ab") as file:
        file.write(b"\x01\x2a\x00")
    restored = ccannpy.Index.load(prefix)
    assert restored.npoints == 1
    assert restored.search(vector, 1)[0][0] == 41
    restored.add(vector + 1, 42)
    del restored
    loaded = ccannpy.Index.load(prefix)
    assert loaded.npoints == 2
    assert set(loaded.search(vector, 2)[0]) == {41, 42}
