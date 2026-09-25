from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import ccannpy


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
