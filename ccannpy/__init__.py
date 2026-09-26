"""Python interface for persistent CCANN indexes."""

import os
import struct
import tempfile
import threading
import zlib

import numpy as np

from ._native import Index as _NativeIndex
from ._native import Metric


_MAGIC = b"CCANNF01"
_HEADER = struct.Struct("<8sI")
_RECORD_HEAD = struct.Struct("<BI")
_CHECKSUM = struct.Struct("<I")
_MERGE_THRESHOLD = 10000
_BINARY = getattr(os, "O_BINARY", 0)


def _sync_directory(path):
    if os.name == "nt":
        return
    directory = os.open(os.path.dirname(path) or ".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_all(fd, data):
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written == 0:
            raise OSError("Short index write")
        offset += written


class _FlatIndex:
    def __init__(self, fd, dimension, path):
        self._fd = fd
        self._dimension = dimension
        self._path = path
        self._lock = threading.Lock()
        self._vectors = {}
        self._seen_tags = set()
        self._pending_removals = 0

    @classmethod
    def create(cls, prefix, dimension, vectors=None, tags=None):
        path = prefix + "_ccann.flat"
        if os.path.exists(prefix + "_disk.index") or os.path.exists(prefix + "_ccann.active"):
            raise ValueError("Index already exists at prefix")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | _BINARY, 0o600)
        try:
            _write_all(fd, _HEADER.pack(_MAGIC, dimension))
            result = cls(fd, dimension, path)
            if vectors is not None:
                for tag, vector in zip(tags, vectors):
                    tag = int(tag)
                    body = _RECORD_HEAD.pack(1, tag) + vector.tobytes()
                    _write_all(fd, body + _CHECKSUM.pack(zlib.crc32(body)))
                    result._seen_tags.add(tag)
                    result._vectors[tag] = vector.copy()
            os.fsync(fd)
            _sync_directory(path)
            return result
        except BaseException:
            os.close(fd)
            raise

    @classmethod
    def load(cls, prefix):
        path = prefix + "_ccann.flat"
        fd = os.open(path, os.O_RDWR | _BINARY)
        result = None
        try:
            header = os.read(fd, _HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError("Corrupt empty index header")
            magic, dimension = _HEADER.unpack(header)
            if magic != _MAGIC or dimension == 0:
                raise ValueError("Corrupt empty index header")
            result = cls(fd, dimension, path)
            size = os.fstat(fd).st_size
            record_size = _RECORD_HEAD.size + dimension * 4 + _CHECKSUM.size
            offset = _HEADER.size
            while offset + record_size <= size:
                record = os.read(fd, record_size)
                body, checksum = record[:-4], _CHECKSUM.unpack(record[-4:])[0]
                if len(record) != record_size or zlib.crc32(body) != checksum:
                    if offset + record_size < size:
                        raise ValueError("Corrupt empty index record")
                    break
                operation, tag = _RECORD_HEAD.unpack(body[:_RECORD_HEAD.size])
                if operation == 1 and tag not in result._seen_tags:
                    result._seen_tags.add(tag)
                    result._vectors[tag] = np.frombuffer(body, dtype="<f4", count=dimension,
                                                         offset=_RECORD_HEAD.size).copy()
                elif operation == 2 and tag in result._vectors:
                    del result._vectors[tag]
                    result._pending_removals += 1
                else:
                    raise ValueError("Corrupt empty index record")
                offset += record_size
            if offset != size:
                os.ftruncate(fd, offset)
                os.fsync(fd)
            os.lseek(fd, offset, os.SEEK_SET)
            if result._pending_removals >= _MERGE_THRESHOLD:
                result._compact_locked()
            return result
        except BaseException:
            if result is not None:
                fd = result._fd
                result._fd = None
            os.close(fd)
            raise

    def _check_vector(self, vector):
        if (not isinstance(vector, np.ndarray) or vector.dtype != np.float32 or
                vector.ndim != 1 or vector.shape[0] != self._dimension or
                not vector.flags.c_contiguous):
            raise ValueError("Vector dimension or dtype does not match the index")

    def _append(self, operation, tag, vector):
        body = _RECORD_HEAD.pack(operation, tag) + vector.tobytes()
        start = os.lseek(self._fd, 0, os.SEEK_END)
        try:
            _write_all(self._fd, body + _CHECKSUM.pack(zlib.crc32(body)))
            os.fsync(self._fd)
        except BaseException:
            os.ftruncate(self._fd, start)
            os.fsync(self._fd)
            raise

    def add(self, vector, tag):
        self._check_vector(vector)
        if not 0 <= tag <= 0xFFFFFFFF:
            raise ValueError("Tag must be uint32")
        with self._lock:
            if tag in self._seen_tags:
                raise ValueError("Tag already exists")
            self._append(1, tag, vector)
            self._seen_tags.add(tag)
            self._vectors[tag] = vector.copy()

    def search(self, query, k, search_l=64):
        self._check_vector(query)
        if not 0 < k <= search_l <= 4096:
            raise ValueError("Require 0 < k <= search_l <= 4096")
        with self._lock:
            if not self._vectors:
                return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.float32)
            tags = np.fromiter(self._vectors, dtype=np.uint32)
            vectors = np.stack(tuple(self._vectors.values()))
        distances = np.sum((vectors - query) ** 2, axis=1)
        nearest = np.argsort(distances, kind="stable")[:k]
        return tags[nearest], distances[nearest].astype(np.float32)

    def remove(self, tag):
        with self._lock:
            if tag not in self._seen_tags:
                raise ValueError("Tag does not exist")
            if tag in self._vectors:
                self._append(2, tag, np.zeros(self._dimension, dtype=np.float32))
                del self._vectors[tag]
                self._pending_removals += 1
                if self._pending_removals >= _MERGE_THRESHOLD:
                    self._compact_locked()

    def merge(self):
        with self._lock:
            if self._pending_removals:
                self._compact_locked()

    def merge_to(self, output_prefix):
        with self._lock:
            path = output_prefix + "_ccann.flat"
            if (os.path.exists(path) or os.path.exists(output_prefix + "_disk.index") or
                    os.path.exists(output_prefix + "_ccann.active")):
                raise ValueError("Merge output prefix already exists")
            directory = os.path.dirname(path) or "."
            fd, temporary = tempfile.mkstemp(prefix=".ccann-merge-", dir=directory)
            try:
                self._write_live_records(fd)
                os.fsync(fd)
                os.close(fd)
                fd = None
                os.link(temporary, path)
                _sync_directory(path)
            finally:
                if fd is not None:
                    os.close(fd)
                os.unlink(temporary)

    def _write_live_records(self, fd):
        _write_all(fd, _HEADER.pack(_MAGIC, self._dimension))
        for tag, vector in self._vectors.items():
            body = _RECORD_HEAD.pack(1, tag) + vector.tobytes()
            _write_all(fd, body + _CHECKSUM.pack(zlib.crc32(body)))

    def _compact_locked(self):
        directory = os.path.dirname(self._path) or "."
        fd, temporary = tempfile.mkstemp(prefix=".ccann-merge-", dir=directory)
        published = False
        try:
            self._write_live_records(fd)
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.close(self._fd)
            self._fd = None
            os.replace(temporary, self._path)
            published = True
            self._fd = os.open(self._path, os.O_RDWR | _BINARY)
            os.lseek(self._fd, 0, os.SEEK_END)
            self._seen_tags = set(self._vectors)
            self._pending_removals = 0
            _sync_directory(self._path)
        finally:
            if self._fd is None:
                self._fd = os.open(self._path, os.O_RDWR | _BINARY)
                os.lseek(self._fd, 0, os.SEEK_END)
            if fd is not None:
                os.close(fd)
            if not published:
                os.unlink(temporary)

    @property
    def npoints(self):
        with self._lock:
            return len(self._vectors)

    @property
    def dimension(self):
        return self._dimension

    def __del__(self):
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None


class Index:
    def __init__(self, backend, threads=4):
        self._backend = backend
        self._threads = threads

    @classmethod
    def create(cls, prefix, vectors, tags, metric=Metric.L2, threads=4):
        if not threads or metric != Metric.L2:
            raise ValueError("Only L2 and positive threads are supported")
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2 and vectors.shape[0] == 0:
            if (vectors.dtype != np.float32 or vectors.shape[1] == 0 or
                    not isinstance(tags, np.ndarray) or tags.dtype != np.uint32 or
                    tags.shape != (0,)):
                raise ValueError("Expected empty float32 vectors and uint32 tags")
            return cls(_FlatIndex.create(prefix, vectors.shape[1]), threads)
        if os.path.exists(prefix + "_ccann.flat") or os.path.exists(prefix + "_ccann.active"):
            raise ValueError("Index already exists at prefix")
        return cls(_NativeIndex.create(prefix, vectors, tags, metric, threads), threads)

    @classmethod
    def load(cls, prefix, metric=Metric.L2, threads=4):
        if os.path.exists(prefix + "_ccann.flat"):
            if not threads or metric != Metric.L2:
                raise ValueError("Only L2 and positive threads are supported")
            return cls(_FlatIndex.load(prefix), threads)
        return cls(_NativeIndex.load(prefix, metric, threads), threads)

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def merge(self, output_prefix=None):
        if output_prefix is None:
            self._backend.merge()
            return self
        self._backend.merge_to(output_prefix)
        return self.load(output_prefix, threads=self._threads)


__all__ = ["Index", "Metric"]
