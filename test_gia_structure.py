"""
Quick end-to-end structural validation for the image-template GIA exporter.

This script generates a tiny image-mode GIA file, prints it with the local
printer, and asserts that the key fields documented in `gia/skills.md` and
`gia/image_template_printed.txt` are still present.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str((ROOT / "gia").resolve()))

import json_to_gia
from print_gia_proto import extract_payload, parse_gia_payload, ProtoPrinter


def build_sample():
    return {
        "elements": [
            {
                "type": "ellipse",
                "relative": {"x": -1.25, "y": 0.75},
                "size": {"rx": 0.5, "ry": 0.5},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "color": "#ff8844",
                "alpha": 0.9,
                "image_asset_ref": 100002,
            },
            {
                "type": "rectangle",
                "relative": {"x": 1.1, "y": -0.6},
                "size": {"width": 1.6, "height": 0.9},
                "rotation": {"x": 0, "y": 0, "z": 22.0},
                "color": "#4fb3ff",
                "alpha": 0.7,
                "image_asset_ref": 100002,
            },
            {
                "type": "triangle",
                "relative": {"x": 0.0, "y": 1.4},
                "size": {"width": 1.2, "height": 1.0392},
                "rotation": {"x": 0, "y": 0, "z": -15.0},
                "color": "#d8a4ff",
                "alpha": 0.8,
                "image_asset_ref": 100002,
            },
        ],
        "mask": {
            "enabled": True,
            "shape_type": "rectangle",
            "center": {"x": 0.0, "y": 0.0},
            "size": {"width": 4.0, "height": 3.0},
        },
    }


def main():
    gia_path = ROOT / "gia" / "test_output_image.gia"
    printed_path = ROOT / "gia" / "test_output_image_printed.txt"

    gia_bytes = json_to_gia.convert_json_to_gia_bytes(
        build_sample(),
        str(ROOT / "gia" / "image_template.gia"),
        mode=json_to_gia.MODE_IMAGE,
    )
    gia_path.write_bytes(gia_bytes)

    printer = ProtoPrinter()
    parse_gia_payload(printer, extract_payload(gia_bytes))
    printed = printer.render()
    printed_path.write_text(printed, encoding="utf-8")

    required_snippets = [
        "kind: 8",
        "resource_class: 15",
        "mask_settings_component: <empty>",
        "image_settings_component: <empty>",
        "field502: 56",
        "field502: 38",
        "shape_type: 1  # rectangle",
        "field503: 1",
    ]

    missing = [snippet for snippet in required_snippets if snippet not in printed]
    if missing:
        raise SystemExit(f"Missing expected snippets: {missing}")

    print("GIA structure validation passed.")
    print(f"Wrote: {gia_path}")
    print(f"Wrote: {printed_path}")


if __name__ == "__main__":
    main()
