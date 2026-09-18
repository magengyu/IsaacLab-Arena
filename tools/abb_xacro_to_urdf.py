#!/usr/bin/env python3
"""Expand a simple ROS-Industrial ABB xacro into a plain URDF.

This is intentionally small and ABB-package-specific. The ABB support packages
used here rely on includes, one robot macro, color properties, material macros,
prefix substitution, and radians(...) expressions, so we avoid requiring a full
ROS/xacro installation for this conversion step.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


XACRO_NS = "{http://ros.org/wiki/xacro}"


def _load_color_properties(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    colors: dict[str, str] = {}
    for prop in root.findall(f"{XACRO_NS}property"):
        name = prop.attrib["name"]
        value = prop.attrib["value"]
        colors[name] = _eval_simple_color(value)
    return colors


def _eval_simple_color(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"{eval(match.group(1), {'__builtins__': {}}, {}):.7f}"

    return re.sub(r"\$\{\s*([^}]+?)\s*\}", repl, value)


def _material_map(colors: dict[str, str]) -> dict[str, str]:
    return {
        name.removeprefix("colour_"): value
        for name, value in colors.items()
        if name.startswith("colour_")
    }


def _entry_info(xacro_path: Path) -> tuple[str, str, Path]:
    root = ET.parse(xacro_path).getroot()
    robot_name = root.attrib["name"]
    include = root.find(f"{XACRO_NS}include")
    if include is None:
        raise RuntimeError(f"No xacro:include found in {xacro_path}")
    include_filename = include.attrib["filename"]
    macro_file = _resolve_find_expr(include_filename, xacro_path)
    macro_call = next(
        child
        for child in list(root)
        if isinstance(child.tag, str) and child.tag.startswith(XACRO_NS) and child.tag != f"{XACRO_NS}include"
    )
    macro_name = macro_call.tag[len(XACRO_NS) :]
    return robot_name, macro_name, macro_file


def _resolve_find_expr(value: str, current_file: Path) -> Path:
    match = re.fullmatch(r"\$\(find ([^)]+)\)/(.*)", value)
    if not match:
        return (current_file.parent / value).resolve()
    package, suffix = match.groups()
    abb_root = current_file.parents[2]
    return (abb_root / package / suffix).resolve()


def _expand_macro(macro_path: Path, robot_name: str, macro_name: str, colors: dict[str, str]) -> ET.Element:
    ET.register_namespace("xacro", "http://ros.org/wiki/xacro")
    root = ET.parse(macro_path).getroot()
    macro = next(
        (child for child in root.findall(f"{XACRO_NS}macro") if child.attrib.get("name") == macro_name),
        None,
    )
    if macro is None:
        raise RuntimeError(f"No xacro:macro named {macro_name!r} found in {macro_path}")

    robot = ET.Element("robot", {"name": robot_name})
    materials = _material_map(colors)

    for child in list(macro):
        tag = child.tag
        if isinstance(tag, str) and tag.startswith(XACRO_NS):
            continue
        robot.append(_convert_element(child, materials))

    return robot


def _convert_element(elem: ET.Element, materials: dict[str, str]) -> ET.Element:
    tag = elem.tag
    if isinstance(tag, str) and tag.startswith(XACRO_NS):
        material_name = tag[len(XACRO_NS) :]
        if material_name.startswith("material_"):
            key = material_name.removeprefix("material_")
            material = ET.Element("material", {"name": key})
            ET.SubElement(material, "color", {"rgba": materials[key]})
            return material
        raise RuntimeError(f"Unsupported xacro element: {tag}")

    converted = ET.Element(tag, {k: _expand_value(v) for k, v in elem.attrib.items()})
    converted.text = elem.text
    converted.tail = elem.tail
    for child in list(elem):
        converted.append(_convert_element(child, materials))
    return converted


def _expand_value(value: str) -> str:
    value = value.replace("${prefix}", "")
    value = re.sub(r"\$\{radians\(([^}]+)\)\}", lambda m: str(_radians(m.group(1))), value)
    if "${" in value:
        raise RuntimeError(f"Unsupported xacro expression in value: {value}")
    return value


def _radians(expr: str) -> float:
    return math.radians(float(eval(expr, {"__builtins__": {}}, {})))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a plain URDF from a simple ROS-Industrial ABB xacro.")
    parser.add_argument(
        "xacro",
        type=Path,
        nargs="?",
        default=Path("references/abb/abb_irb120_support/urdf/irb120_3_58.xacro"),
        help="Input ABB xacro entry file.",
    )
    parser.add_argument(
        "--abb-root",
        type=Path,
        default=Path("references/abb"),
        help="Path to the cloned ros-industrial/abb repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output URDF path. Defaults to input xacro path with .urdf suffix.",
    )
    args = parser.parse_args()

    xacro_path = args.xacro.resolve()
    output = args.output or xacro_path.with_suffix(".urdf")
    abb_root = args.abb_root.resolve()
    colors = _load_color_properties(abb_root / "abb_resources/urdf/common_colours.xacro")
    robot_name, macro_name, macro_path = _entry_info(xacro_path)
    robot = _expand_macro(macro_path, robot_name, macro_name, colors)

    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
