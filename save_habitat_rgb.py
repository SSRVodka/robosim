#!/usr/bin/env python3
"""Save one frame from the Habitat RGB camera."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from control_stubs.tools.client import RobosimClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--sensor", default="habitat_rgb")
    parser.add_argument("--output", default="habitat_rgb.png")
    return parser


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        Image.fromarray(rgb, "RGB").save(path)
        return

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Saving PNG requires pillow or opencv-python") from exc

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> int:
    args = build_parser().parse_args()
    client = RobosimClient(args.host, args.port)
    try:
        data = client.sensing.get_sensors([args.sensor])
        if not data.images:
            raise RuntimeError(f"No image returned for sensor {args.sensor!r}")

        image = data.images[0]
        rgb = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height,
            image.width,
            3,
        )
        output = Path(args.output)
        save_rgb(output, rgb)
        print(f"saved {output} {image.width}x{image.height} {image.encoding}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
