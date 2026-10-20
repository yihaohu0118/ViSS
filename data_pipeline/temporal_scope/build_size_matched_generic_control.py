#!/usr/bin/env python3
"""Select a deterministic, source-stratified generic temporal SFT control."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SUPPORTED_PARENT_KINDS = (
    "source_semantic_row",
    "cited_nonbenchmark_refresh_projection",
)


def _rank(seed: str, parent_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{parent_id}".encode()).hexdigest()


def select_parent_ids(
    rows: list[dict[str, Any]], *, seed: str, quotas: dict[str, int]
) -> tuple[list[str], dict[str, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        parents = row.get("parents") or []
        if len(parents) != 1:
            raise ValueError("generic control requires exactly one parent per row")
        parent_id = str(parents[0].get("id") or "")
        parent_kind = str(parents[0].get("kind") or "")
        if not parent_id or parent_kind not in SUPPORTED_PARENT_KINDS:
            raise ValueError("generic control parent provenance is malformed")
        if parent_id in seen:
            raise ValueError("generic control parent IDs must be unique")
        seen.add(parent_id)
        grouped[parent_kind].append(parent_id)

    selected: list[str] = []
    counts: dict[str, int] = {}
    for kind in SUPPORTED_PARENT_KINDS:
        quota = int(quotas.get(kind, 0))
        candidates = grouped[kind]
        if quota <= 0 or quota > len(candidates):
            raise ValueError(f"invalid quota for {kind}: {quota}/{len(candidates)}")
        ranked = sorted(candidates, key=lambda value: (_rank(seed, value), value))
        chosen = ranked[:quota]
        selected.extend(chosen)
        counts[kind] = len(chosen)
    return sorted(selected), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="viss-generic-control-7677-v1")
    parser.add_argument("--source-semantic-rows", type=int, default=5903)
    parser.add_argument("--cited-refresh-rows", type=int, default=1774)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    trainer = source / "trainer/sft.jsonl"
    manifest = source / "manifest.json"
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in trainer.read_text().splitlines() if line]
    selected, counts = select_parent_ids(
        rows,
        seed=args.seed,
        quotas={
            "source_semantic_row": args.source_semantic_rows,
            "cited_nonbenchmark_refresh_projection": args.cited_refresh_rows,
        },
    )
    if len(selected) != args.source_semantic_rows + args.cited_refresh_rows:
        raise AssertionError("generic control selection size diverged")

    payload = {
        "format": "iclr_video.capability_parent_selection.v1",
        "status": "sealed_deduplicated_union",
        "purpose": "size_matched_generic_temporal_sft_control",
        "seed": args.seed,
        "source_release": str(source),
        "source_release_id": json.loads(manifest.read_text()).get("release_id"),
        "parent_ids": selected,
        "profiles": list(SUPPORTED_PARENT_KINDS),
        "counts_by_profile": counts,
        "total_rows": len(selected),
        "contracts": {
            "benchmark_content_used_for_selection": False,
            "source_release_membership_verified": True,
            "model_predictions_used_for_selection": False,
            "task_labels_used_for_selection": False,
            "selection_is_sha256_ranked_within_source_kind": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(selected), "counts": counts}))


if __name__ == "__main__":
    main()
