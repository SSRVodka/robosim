"""Compile a CSD package into a backend realization from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robosim.core import compile_csd


def main() -> None:
    """Run one CSD realization and print its manifest or blockers as JSON."""
    parser = argparse.ArgumentParser(description="Compile an OpenUSD CSD package.")
    parser.add_argument("--backend", choices=("mujoco", "pybullet"), default="mujoco")
    parser.add_argument("--csd", required=True, type=Path, help="Path to package scene.usda")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Destination engine_manifests directory (default: package/engine_manifests)",
    )
    parser.add_argument(
        "--realization-config",
        default="{}",
        help="JSON object included in the realization cache key",
    )
    parser.add_argument("--realization-version", default="csd-compiler-0.10")
    args = parser.parse_args()
    try:
        config = json.loads(args.realization_config)
    except json.JSONDecodeError as error:
        parser.error(f"invalid --realization-config JSON: {error.msg}")
    if not isinstance(config, dict):
        parser.error("--realization-config must be a JSON object")
    output_root = args.output_root or args.csd.parent / "engine_manifests"
    result = compile_csd(
        backend=args.backend,
        csd_path=args.csd,
        output_root=output_root,
        realization_config=config,
        realization_version=args.realization_version,
    )
    if result.manifest is not None:
        print(json.dumps(result.manifest.to_json_dict(), indent=2, sort_keys=True))
        return
    print(json.dumps([blocker.to_json_dict() for blocker in result.blockers], indent=2))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
