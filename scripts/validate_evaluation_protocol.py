#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from viss_core.evaluation_protocol import LOCKED_PROTOCOL, validate_locked_protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the immutable exports and rules of the locked evaluation protocol."
    )
    parser.add_argument("--protocol", type=Path, default=LOCKED_PROTOCOL)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_locked_protocol(args.protocol),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
