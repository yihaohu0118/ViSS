"""Bind the public protocol template to the user's own evaluation assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.evaluate_qwen3vl_recent4_full import (
    sha256_file,
    validate_processor_contract,
)


def configure(
    *,
    template: Path,
    streaming_samples: Path,
    ovo_samples: Path,
    processor_config: Path,
    output: Path,
) -> None:
    protocol = json.loads(template.read_text(encoding="utf-8"))
    for benchmark, path in (
        ("streamingbench", streaming_samples),
        ("ovo", ovo_samples),
    ):
        path = path.resolve(strict=True)
        protocol["benchmarks"][benchmark]["export"] = str(path)
        protocol["benchmarks"][benchmark]["export_sha256"] = sha256_file(path)
    processor_config = processor_config.resolve(strict=True)
    if processor_config.name != "preprocessor_config.json":
        raise ValueError("Pass the processor's preprocessor_config.json")
    protocol["inference"]["processor_contract"]["preprocessor_config_sha256"] = (
        sha256_file(processor_config)
    )
    validate_processor_contract(processor_config.parent, protocol)
    protocol["status"] = "locked"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/evaluation/recent4.example.json"),
    )
    parser.add_argument("--streaming-samples", type=Path, required=True)
    parser.add_argument("--ovo-samples", type=Path, required=True)
    parser.add_argument("--processor-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure(**vars(args))
    print(f"Created local evaluation protocol: {args.output}")


if __name__ == "__main__":
    main()
