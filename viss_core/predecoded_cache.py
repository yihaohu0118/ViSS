"""Indexed, lossless recent-frame cache for CPU/GPU pipeline separation."""

from __future__ import annotations

import copy
import hashlib
import json
import mmap
from pathlib import Path
from typing import Any


FORMAT = "iclr_video.predecoded_recent4_mmap.v1"
FORMAT_V2 = "iclr_video.predecoded_causal_view_mmap.v2"
FORMAT_V3 = "iclr_video.predecoded_mixed_resolution_mmap.v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PredecodedRecent4Cache:
    """Read exact RGB frames from one mmap-backed, benchmark-wide cache."""

    def __init__(
        self,
        root: Path,
        *,
        benchmark: str,
        samples_sha256: str,
        annotation_sha256: str,
        wanted: list[str],
        frame_policy: str = "recent4",
        visual_input_mode: str = "image-list",
    ) -> None:
        root = root.resolve()
        marker = root / "DATA_READY"
        index_path = root / "index.json"
        if not marker.is_file() or not index_path.is_file():
            raise ValueError(f"predecoded cache is not DATA_READY: {root}")
        ready = json.loads(marker.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if ready.get("format") not in {FORMAT, FORMAT_V2, FORMAT_V3} or index.get(
            "format"
        ) != ready.get("format"):
            raise ValueError("predecoded cache format mismatch")
        if ready.get("status") != "complete":
            raise ValueError("predecoded cache marker is not complete")
        if sha256_file(index_path) != ready.get("index_sha256"):
            raise ValueError("predecoded cache index checksum mismatch")
        for key, expected in (
            ("benchmark", benchmark),
            ("samples_sha256", samples_sha256),
            ("annotation_sha256", annotation_sha256),
            ("frame_policy", frame_policy),
            ("visual_input_mode", visual_input_mode),
        ):
            if index.get(key) != expected:
                raise ValueError(f"predecoded cache {key} mismatch")
        entries = index.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("predecoded cache entries are invalid")
        missing = [uid for uid in wanted if uid not in entries]
        if missing:
            raise ValueError(f"predecoded cache misses requested UIDs: {missing[:20]}")
        frames_path = root / str(index.get("frames_file") or "frames.bin")
        if not frames_path.is_file():
            raise ValueError("predecoded cache frame store is missing")
        if sha256_file(frames_path) != index.get("frames_sha256"):
            raise ValueError("predecoded cache frame checksum mismatch")
        self.root = root
        self.index = index
        self.entries = entries
        self._handle = frames_path.open("rb")
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "format": self.index["format"],
            "root": str(self.root),
            "benchmark": self.index["benchmark"],
            "num_entries": int(self.index["num_entries"]),
            "frames_sha256": self.index["frames_sha256"],
        }

    def get(
        self, uid: str, *, video_path: Path, query_time: float
    ) -> tuple[list[Any], dict[str, Any]]:
        import numpy as np
        from PIL import Image

        entry = self.entries.get(uid)
        if not isinstance(entry, dict):
            raise KeyError(uid)
        if Path(str(entry["video"])).name != video_path.name:
            raise ValueError(f"predecoded cache video mismatch for {uid}")
        if float(entry["query_time"]).hex() != float(query_time).hex():
            raise ValueError(f"predecoded cache query time mismatch for {uid}")
        frame_records = entry.get("frames")
        if frame_records is not None:
            if not isinstance(frame_records, list) or not frame_records:
                raise ValueError(f"predecoded mixed-resolution frames are invalid for {uid}")
            frames = []
            for record in frame_records:
                shape = tuple(int(value) for value in record["shape"])
                if len(shape) != 3 or min(shape) <= 0 or shape[-1] != 3:
                    raise ValueError(f"predecoded cache frame shape is invalid for {uid}")
                offset = int(record["offset"])
                nbytes = int(record["nbytes"])
                expected_bytes = int(np.prod(shape))
                if (
                    nbytes != expected_bytes
                    or offset < 0
                    or offset + nbytes > len(self._mmap)
                ):
                    raise ValueError(f"predecoded cache byte range is invalid for {uid}")
                array = np.ndarray(
                    shape, dtype=np.uint8, buffer=self._mmap, offset=offset
                )
                frames.append(Image.fromarray(array.copy(), mode="RGB"))
        else:
            shape = tuple(int(value) for value in entry["shape"])
            if len(shape) != 4 or shape[0] <= 0 or shape[-1] != 3:
                raise ValueError(f"predecoded cache shape is invalid for {uid}")
            offset = int(entry["offset"])
            nbytes = int(entry["nbytes"])
            expected_bytes = int(np.prod(shape))
            if nbytes != expected_bytes or offset < 0 or offset + nbytes > len(self._mmap):
                raise ValueError(f"predecoded cache byte range is invalid for {uid}")
            array = np.ndarray(shape, dtype=np.uint8, buffer=self._mmap, offset=offset)
            frames = [Image.fromarray(frame, mode="RGB") for frame in array]
        sampling = copy.deepcopy(entry["sampling"])
        sampling["source_decode_backend"] = sampling.get("decode_backend")
        sampling["decode_backend"] = (
            "predecoded_recent4_mmap_v1"
            if self.index["frame_policy"] == "recent4"
            else f"predecoded_{self.index['frame_policy']}_mmap_v2"
        )
        sampling["predecoded_cache_hit"] = True
        return frames, sampling

    def close(self) -> None:
        self._mmap.close()
        self._handle.close()
