import os
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
    assert not hasattr(index, "save")
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
    assert not hasattr(index, "save")
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
import os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ccannpy
index = ccannpy.Index.load(sys.argv[1], threads=2)
index.add(np.full(64, 7, dtype=np.float32), 9001)
vectors = np.eye(8, 64, dtype=np.float32) * 20
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(lambda item: index.add(*item), zip(vectors, range(9010, 9018))))
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", script, prefix], check=False, timeout=20)
    assert result.returncode == 17
    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == len(vectors) + 9
    inserted = [(9001, np.full(64, 7, dtype=np.float32))]
    inserted.extend((9010 + offset, vector) for offset, vector in
                    enumerate(np.eye(8, 64, dtype=np.float32) * 20))
    if empty:
        for tag, vector in inserted:
            ids, distances = restored.search(vector, 1)
            assert ids[0] == tag
            assert distances[0] == 0
    else:
        metadata = np.fromfile(prefix + "_disk.index", dtype=np.uint64, count=5, offset=8)
        node_size, nodes_per_sector = map(int, metadata[3:5])
        count = int(np.fromfile(prefix + "_disk.index.tags", dtype=np.uint32, count=1)[0])
        persisted_tags = np.fromfile(prefix + "_disk.index.tags", dtype=np.uint32,
                                     count=count, offset=8)
        locations = np.fromfile(prefix + "_disk.index.id2loc", dtype=np.uint32, count=count)
        assert count == len(vectors) + len(inserted)
        for tag, vector in inserted:
            matches = np.flatnonzero(persisted_tags == tag)
            assert len(matches) == 1
            location = int(locations[matches[0]])
            node_offset = ((1 + location // nodes_per_sector) * 4096 +
                           location % nodes_per_sector * node_size)
            persisted_vector = np.fromfile(prefix + "_disk.index", dtype=np.float32,
                                           count=64, offset=node_offset)
            np.testing.assert_array_equal(persisted_vector, vector)


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


def test_graph_remove_appends_log_and_repairs_incomplete_tail(ssd_index):
    prefix, vectors, tags = ssd_index
    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    log = prefix + "_ccann.removed"
    index.remove(1000)
    first = open(log, "rb").read()
    assert first == b"1000\n"
    index.remove(1001)
    assert open(log, "rb").read() == first + b"1001\n"
    index.remove(1001)
    assert open(log, "rb").read() == first + b"1001\n"
    del index

    with open(log, "ab") as file:
        file.write(b"1002")
    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 298
    restored.remove(1003)
    assert open(log, "rb").read() == b"1000\n1001\n1003\n"
    del restored
    assert ccannpy.Index.load(prefix, threads=2).npoints == 297


def test_graph_remove_survives_process_kill(ssd_index, tmp_path):
    prefix, vectors, tags = ssd_index
    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    del index

    script = """
import os, sys
import ccannpy
index = ccannpy.Index.load(sys.argv[1], threads=2)
index.remove(1000)
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", script, prefix], check=False, timeout=20)
    assert result.returncode == 17
    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 299
    restored.remove(1001)
    restored.add(np.full(64, 7, dtype=np.float32), 9001)
    assert hasattr(restored, "merge")
    assert not (tmp_path / "index_ccann.active").exists()
    del restored
    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 299
    assert restored.search(np.full(64, 7, dtype=np.float32), 1)[0][0] == 9001


def test_empty_index_auto_compacts_deleted_records(tmp_path, monkeypatch):
    assert ccannpy._MERGE_THRESHOLD == 10000
    monkeypatch.setattr(ccannpy, "_MERGE_THRESHOLD", 3)
    prefix = str(tmp_path / "empty")
    index = ccannpy.Index.create(prefix, np.empty((0, 4), dtype=np.float32),
                                 np.empty(0, dtype=np.uint32))
    for tag in range(5):
        index.add(np.full(4, tag, dtype=np.float32), tag)
    assert hasattr(index, "merge")
    index.remove(1)
    index.remove(3)
    before = (tmp_path / "empty_ccann.flat").stat().st_size
    index.remove(4)
    assert index.npoints == 2
    assert (tmp_path / "empty_ccann.flat").stat().st_size < before
    assert ccannpy.Index.load(prefix).npoints == 2
    index.add(np.full(4, 6, dtype=np.float32), 6)
    del index
    assert set(ccannpy.Index.load(prefix).search(np.zeros(4, dtype=np.float32), 3)[0]) == {0, 2, 6}


def test_graph_auto_merge_at_10000_deletes(tmp_path):
    prefix = str(tmp_path / "index")
    vectors = np.random.default_rng(31).random((10001, 32), dtype=np.float32)
    tags = np.arange(10001, dtype=np.uint32)
    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    assert hasattr(index, "merge")
    del index
    script = """
import os, sys
import ccannpy
index = ccannpy.Index.load(sys.argv[1], threads=2)
for tag in range(9999):
    index.remove(tag)
assert index.npoints == 2
assert not os.path.exists(sys.argv[1] + "_ccann.active")
index.remove(9999)
assert index.npoints == 1
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", script, prefix], check=False, timeout=90)
    assert result.returncode == 17
    assert (tmp_path / "index_ccann.active").read_text() == "1\n"
    assert not (tmp_path / "index_disk.index").exists()
    assert not (tmp_path / "index_ccann.removed").exists()
    with pytest.raises(ValueError):
        ccannpy.Index.create(prefix, vectors, tags, threads=2)

    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 1
    assert restored.search(vectors[10000], 1)[0][0] == 10000
    restored.add(np.full(32, 9, dtype=np.float32), 20000)
    del restored
    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 2
    with open(prefix + "_ccann_gen_1_disk.index.tags", "rb") as file:
        count, width = np.fromfile(file, dtype=np.uint32, count=2)
        persisted = np.fromfile(file, dtype=np.uint32, count=int(count))
    assert width == 1
    assert set(persisted) == {10000, 20000}


def test_graph_all_deleted_at_threshold_can_accept_new_add(tmp_path):
    prefix = str(tmp_path / "index")
    vectors = np.random.default_rng(32).random((10000, 32), dtype=np.float32)
    tags = np.arange(10000, dtype=np.uint32)
    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    for tag in range(9999):
        index.remove(tag)
    assert index.npoints == 1
    del index

    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 1
    restored.remove(9999)
    assert restored.npoints == 0
    del restored
    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 0
    restored.add(np.full(32, 7, dtype=np.float32), 20000)
    assert restored.npoints == 1
    assert (tmp_path / "index_ccann.active").read_text() == "1\n"
    del restored
    loaded = ccannpy.Index.load(prefix, threads=2)
    assert loaded.npoints == 1
    with open(prefix + "_ccann_gen_1_disk.index.tags", "rb") as file:
        count, _ = np.fromfile(file, dtype=np.uint32, count=2)
        persisted = np.fromfile(file, dtype=np.uint32, count=int(count))
    assert count == 1
    assert persisted[0] == 20000


def test_graph_load_finishes_interrupted_auto_merge(tmp_path):
    prefix = str(tmp_path / "index")
    vectors = np.random.default_rng(33).random((10001, 32), dtype=np.float32)
    tags = np.arange(10001, dtype=np.uint32)
    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    for tag in range(9999):
        index.remove(tag)
    del index

    log_fd = os.open(prefix + "_ccann.removed", os.O_WRONLY | os.O_APPEND)
    try:
        os.write(log_fd, b"9999\n")
        os.fsync(log_fd)
    finally:
        os.close(log_fd)
    restored = ccannpy.Index.load(prefix, threads=2)
    assert restored.npoints == 1
    assert (tmp_path / "index_ccann.active").read_text() == "1\n"
    assert restored.search(vectors[10000], 1)[0][0] == 10000


def test_graph_manual_merge_and_snapshot(ssd_index, tmp_path):
    prefix, vectors, tags = ssd_index
    index = ccannpy.Index.create(prefix, vectors, tags, threads=2)
    index.remove(1000)
    assert index.merge() is index
    assert (tmp_path / "index_ccann.active").read_text() == "1\n"
    assert not (tmp_path / "index_ccann_gen_1_ccann.removed").exists()
    assert index.npoints == 299
    assert index.merge() is index
    assert (tmp_path / "index_ccann.active").read_text() == "2\n"
    index.add(np.full(64, 7, dtype=np.float32), 9001)
    index.remove(1001)

    snapshot_prefix = str(tmp_path / "snapshot")
    snapshot = index.merge(snapshot_prefix)
    with pytest.raises(ValueError, match="already exists"):
        index.merge(snapshot_prefix)
    assert snapshot.npoints == index.npoints == 299
    assert not (tmp_path / "snapshot_ccann.active").exists()
    with open(snapshot_prefix + "_disk.index.tags", "rb") as file:
        count, _ = np.fromfile(file, dtype=np.uint32, count=2)
        snapshot_tags = np.fromfile(file, dtype=np.uint32, count=int(count))
    assert 1000 not in snapshot_tags and 1001 not in snapshot_tags
    assert 9001 in snapshot_tags
    index.remove(1002)
    assert index.npoints == 298
    assert snapshot.npoints == 299
    assert ccannpy.Index.load(prefix, threads=2).npoints == 298
    assert ccannpy.Index.load(snapshot_prefix, threads=2).npoints == 299


def test_empty_index_manual_merge_and_snapshot(tmp_path):
    prefix = str(tmp_path / "empty")
    index = ccannpy.Index.create(prefix, np.empty((0, 4), dtype=np.float32),
                                 np.empty(0, dtype=np.uint32))
    vector = np.array([1, 2, 3, 4], dtype=np.float32)
    index.add(vector, 41)
    index.add(vector + 1, 42)
    index.remove(42)
    before = (tmp_path / "empty_ccann.flat").stat().st_size
    assert index.merge() is index
    assert (tmp_path / "empty_ccann.flat").stat().st_size < before
    snapshot_prefix = str(tmp_path / "snapshot")
    snapshot = index.merge(snapshot_prefix)
    assert snapshot.npoints == 1
    index.add(vector + 2, 43)
    assert index.npoints == 2
    assert ccannpy.Index.load(snapshot_prefix).npoints == 1
    assert ccannpy.Index.load(prefix).npoints == 2
