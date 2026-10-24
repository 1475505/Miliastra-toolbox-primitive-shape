import copy
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import client_ui_export
import gia_lua as wire  # 独立读取器；不使用导出器的编码辅助函数。
import server


def sample_result(mask=False):
    return {
        "mode": "fill", "elements_count": 3, "elapsed_seconds": .01,
        "image_size": {"width": 200, "height": 160},
        "image_center": {"x": 100, "y": 80},
        "config": {"pixel_per_unit": 2},
        "elements": [
            {"type": "rect", "center": {"x": 50, "y": -40},
             "size": {"width": 104, "height": 84}, "color": "#ffffff", "alpha": 1,
             "is_background": True},
            {"type": "circle", "center": {"x": 60, "y": -30},
             "size": {"rx": 3, "ry": 5}, "rotation": {"x": 5, "y": 90, "z": 30},
             "color": "#123456", "alpha": .5},
            {"type": "triangle", "center": {"x": 40, "y": -50},
             "size": {"width": 12, "height": 18}, "rotation": 45,
             "color": "#abcdef", "alpha": 0},
        ],
        "mask": {"enabled": mask, "shape_type": "rectangle",
                 "center": {"x": 55, "y": -35}, "size": {"width": 70, "height": 50}},
    }


def content(unit):
    return wire._message(wire._message(unit, 19), 1)


def parts(unit):
    return wire._components(content(unit))


def body(unit, kind, tag):
    return wire._message(wire._message(parts(unit)[kind], 503), tag)


def guid(unit):
    return wire._get(wire._message(unit, 1), 4)


def related(unit):
    return [wire._get(value, 4) for value in wire._messages(unit, 2)]


def decode(data):
    root = wire._fields(data[20:-4])
    main = list(wire._messages(root, 1))[0]
    return root, main, [main] + list(wire._messages(root, 2))


def layouts(unit):
    transform = body(unit, 12, 13)
    multi = wire._message(transform, 12)
    return multi, [(wire._get(platform, 501), wire._message(platform, 502))
                   for platform in wire._messages(multi, 501)]


class ClientUIExportTests(unittest.TestCase):
    def test_container_node_template_identity_and_component_markers(self):
        result = sample_result()
        before = copy.deepcopy(result)
        data = client_ui_export.build_container_node_gia(result, "中文拼图")
        self.assertEqual(result, before)
        self.assertEqual(struct.unpack(">5I", data[:20]), (len(data) - 4, 1, 0x326, 3, len(data) - 24))
        self.assertEqual(data[-4:], b"\0\0\x06y")
        root, main, units = decode(data)
        self.assertEqual(len(list(wire._messages(root, 1))), 1)
        self.assertNotIn(4, root)  # 不套用素材组的经典/超限标记。
        self.assertEqual(wire._get(root, 5, 2).decode(), "7.1.0")
        self.assertEqual(wire._get(main, 3, 2).decode(), "中文拼图")
        self.assertEqual(wire._get(main, 5), 70)  # 客户端控件模板，不是服务端容器(21)。
        self.assertEqual(len(units), 4)
        ids = [guid(unit) for unit in units]
        self.assertEqual(ids, list(range(1073741825, 1073741829)))
        self.assertIs(main, units[0])
        self.assertEqual(guid(main), 1073741825)
        self.assertEqual(related(main), ids[1:])
        self.assertEqual(related(units[0]), ids[1:])
        self.assertEqual(wire._children(content(units[0])), ids[1:])
        self.assertIn(91, parts(main))  # 容器节点专用槽。
        self.assertNotIn(82, parts(main))  # 无服务端根节点引用槽。
        self.assertNotIn(504, content(units[0]))
        self.assertEqual(wire._children(content(main)), ids[1:])
        self.assertNotIn(504, content(main))
        for unit in units[1:]:
            self.assertEqual(wire._get(content(unit), 504), ids[0])
            self.assertFalse(related(unit))
            self.assertFalse(wire._children(content(unit)))
        markers = {12: (1, 4), 86: (4, 5), 83: (63, 65),
                   90: (67, 69), 91: (68, 70), 96: (73, 75), 97: (74, 76)}
        asset_indices = []
        for unit in units:
            identity = wire._message(unit, 1)
            self.assertEqual((wire._get(identity, 2), wire._get(identity, 3)), (1, 8))
            meta = content(unit)
            self.assertEqual(wire._get(meta, 501), guid(unit))
            infos = list(wire._messages(meta, 502))
            self.assertEqual(wire._get(wire._message(infos[0], 11), 501), guid(unit))
            self.assertEqual(len(infos), 3 if unit is main else 2)
            asset_indices.append(wire._get(wire._message(infos[1], 12), 501))
            self.assertEqual(wire._get(unit, 5), 70 if unit is main else 15)
            for component in wire._messages(meta, 505):
                kind = wire._get(component, 502)
                if kind == 15:
                    self.assertNotIn(503, component)
                    continue
                details = wire._message(component, 503)
                self.assertEqual((wire._get(component, 501), wire._get(details, 501)), markers[kind])
                self.assertEqual(wire._get(details, 502), kind)
                self.assertEqual(wire._get(details, 503), 1)
                self.assertEqual(wire._message(details, 504), identity)
        self.assertEqual(asset_indices, [1, 2, 3, 4])
        self.assertEqual([wire._get(wire._message(info, 12), 501) for info in
                          wire._messages(content(main), 502) if 12 in info], [1])
        # 新资产不引用来自其他模板的伴生 GUID。
        self.assertFalse(wire._message(list(wire._messages(content(main), 502))[-1], 14))

    def test_four_platform_layout_color_alpha_pivot_and_storage_order(self):
        _, main, units = decode(client_ui_export.build_container_node_gia(sample_result()))
        # 输入按底→顶，客户端 GIA 同方向；不能整体倒序。
        self.assertEqual([wire._get(unit, 3, 2).decode() for unit in units[1:]], ["1", "2", "3"])
        images = [body(unit, 96, 84) for unit in units[1:]]
        self.assertEqual([wire._get(image, 503) for image in images], [100001, 100002, 100003])
        self.assertEqual([wire._get(image, 502) for image in images], [0xFFFFFFFF, 0x80123456, 0x00ABCDEF])
        self.assertEqual([wire._get(image, 504) for image in images], [1, 1, 1])  # Stretch
        for unit in units:
            multi, platforms = layouts(unit)
            self.assertEqual([device for device, _ in platforms], [0, 1, 2, 3])
            self.assertEqual(wire._get(multi, 504), 1)
            self.assertTrue(all(rect == platforms[0][1] for _, rect in platforms))
            rect = platforms[0][1]
            if unit is main:
                self.assertEqual(wire._get(multi, 502), 25)
                self.assertIn(502, rect)
                self.assertEqual(wire._vector(rect, 502), [.5, .5])
                self.assertEqual(wire._vector(rect, 503), [.5, .5])
                self.assertEqual(wire._vector(rect, 505), [200, 160])
                self.assertEqual(wire._vector(rect, 504), [0, 0])
            else:
                self.assertEqual(wire._get(multi, 502), 23)
                self.assertEqual(wire._vector(rect, 502), [.5, .5])
                config = body(unit, 97, 85)
                for flag in (501, 509, 514, 516):
                    self.assertFalse(wire._get(config, flag))
                self.assertIn(84, parts(unit)[97])  # 图片 B 槽显式存在。
        ellipse = layouts(units[2])[1][0][1]
        self.assertEqual(wire._vector(ellipse, 504), [20, 20])
        self.assertEqual(wire._vector(ellipse, 505), [12, 20])
        self.assertEqual(wire._vector(ellipse, 508, (1, 2, 3), (0, 0, 0)), [5, 90, 30])
        triangle = layouts(units[3])[1][0][1]
        self.assertEqual(wire._vector(triangle, 504), [-20, -20])
        self.assertEqual(wire._vector(triangle, 505), [24, 36])
        self.assertAlmostEqual(wire._vector(triangle, 506)[1], 1 / 3)
        self.assertEqual(wire._vector(triangle, 508, (1, 2, 3), (0, 0, 0)), [0, 0, 45])

    def test_rectangular_mask_is_transparent_parent_with_rebased_children(self):
        data = client_ui_export.build_container_node_gia(sample_result(mask=True), origin_x=10, origin_y=15)
        _, main, units = decode(data)
        root, mask, *images = units
        self.assertEqual(len(units), 5)
        self.assertEqual(related(main), [guid(mask)])
        self.assertEqual(wire._children(content(root)), [guid(mask)])
        self.assertEqual(wire._children(content(mask)), [guid(image) for image in images])
        self.assertEqual(wire._get(body(mask, 96, 84), 502), 0x00FFFFFF)
        self.assertEqual(wire._get(body(mask, 97, 85), 501), 1)
        mask_rect = layouts(mask)[1][0][1]
        self.assertEqual(wire._vector(mask_rect, 504), [100, -55])
        self.assertEqual(wire._vector(mask_rect, 505), [140, 100])
        for index, image in enumerate(images):
            self.assertEqual(wire._get(content(image), 504), guid(mask))
            rect = layouts(image)[1][0][1]
            x, y = wire._vector(rect, 504)
            src = sample_result()["elements"][index]["center"]
            self.assertEqual((x + 100, y - 55), (src["x"] * 2 - 10, src["y"] * 2 + 15))

    def test_custom_origin_without_mask_and_custom_asset(self):
        result = sample_result()
        result["elements"][0]["image_asset_ref"] = 123456
        _, _, units = decode(client_ui_export.build_container_node_gia(result, origin_x=10, origin_y=15))
        rect = layouts(units[1])[1][0][1]
        self.assertEqual(wire._vector(rect, 504), [90, -65])
        self.assertEqual(wire._get(body(units[1], 96, 84), 503), 123456)

    def test_alpha_fallback_integer_one_and_short_color(self):
        for overrides, expected in [({"alpha": 1}, 0xFF123456), ({"alpha": 0}, 0x00123456),
                                    ({"alpha": None}, 0x80123456),
                                    ({"alpha": None, "packed_color": 0x33123456}, 0x33123456),
                                    ({"alpha": None, "color": [1, 2, 3, 64]}, 0x40010203),
                                    ({"alpha": .5, "color": "#abc"}, 0x80AABBCC)]:
            with self.subTest(overrides=overrides):
                result = sample_result()
                result["elements"] = [dict(result["elements"][1], **overrides)]
                _, _, units = decode(client_ui_export.build_container_node_gia(result))
                self.assertEqual(wire._get(body(units[1], 96, 84), 502), expected)

    def test_missing_triangle_height_uses_equilateral_height(self):
        result = sample_result()
        result["elements"] = [result["elements"][2]]
        del result["elements"][0]["size"]["height"]
        _, _, units = decode(client_ui_export.build_container_node_gia(result))
        self.assertAlmostEqual(wire._vector(layouts(units[1])[1][0][1], 505)[1], 24 * 3 ** .5 / 2, places=4)

    def test_empty_container_has_no_dangling_children(self):
        result = sample_result()
        result["elements"] = []
        _, main, units = decode(client_ui_export.build_container_node_gia(result))
        self.assertEqual(len(units), 1)
        self.assertEqual(wire._get(main, 5), 70)
        self.assertFalse(related(main))
        self.assertFalse(wire._children(content(units[0])))

    def test_invalid_input_fails_instead_of_silently_dropping_shapes(self):
        changes = [lambda r: r["image_size"].update(width=0),
                   lambda r: r["config"].update(pixel_per_unit=float("nan")),
                   lambda r: r["elements"][0].update(type="star"),
                   lambda r: r["elements"][0]["size"].update(width=0),
                   lambda r: r["elements"][0]["center"].update(x=float("inf")),
                   lambda r: r["elements"][0].update(rotation={"z": float("nan")}),
                   lambda r: r["elements"][0].update(alpha=float("nan")),
                   lambda r: r["elements"][0].update(image_asset_ref=1.5),
                   lambda r: r["mask"].update(enabled=True, shape_type="ellipse")]
        for change in changes:
            result = sample_result()
            change(result)
            with self.subTest(result=result), self.assertRaises(ValueError):
                client_ui_export.build_container_node_gia(result)
        with self.assertRaises(ValueError):
            client_ui_export.build_container_node_gia(sample_result(), origin_x=float("nan"))


class ClientUIIntegrationTests(unittest.TestCase):
    def test_local_result_registration_download_name_and_result_button(self):
        with server.app.test_client() as client:
            response = client.post("/register_result", json={"image_name": "原图.png", "result": sample_result()})
            tid = response.get_json()["task_id"]
            try:
                with mock.patch.object(server, "_load_json_to_gia", side_effect=AssertionError("must not use pyc")):
                    download = client.get(f"/download_container_node_gia/{tid}?export_name=custom&origin_x=10&origin_y=15")
                self.assertEqual(download.status_code, 200)
                self.assertIn("custom_container_node.gia", download.headers["Content-Disposition"])
                self.assertEqual(download.mimetype, "application/octet-stream")
                _, main, units = decode(download.data)
                self.assertEqual(wire._get(main, 3, 2).decode(), "custom")
                self.assertEqual(wire._vector(layouts(units[1])[1][0][1], 504), [90, -65])
                page = client.get(f"/result/{tid}")
                self.assertEqual(page.status_code, 200)
                self.assertIn('id="btnExportContainerNode"', page.get_data(as_text=True))
                self.assertNotIn('id="btnExportClientUI"', page.get_data(as_text=True))
                self.assertEqual(client.get(f"/download_client_ui_gia/{tid}").status_code, 404)
                for value in ("oops", "nan", "inf"):
                    self.assertEqual(client.get(f"/download_container_node_gia/{tid}?origin_x={value}").status_code, 400)
            finally:
                server.tasks.pop(tid, None)
            self.assertEqual(client.get("/download_container_node_gia/nonexistent").status_code, 404)

    def test_cli_exports_container_without_material_group_builder(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("OPENCODE_TEST_TMP")) as directory:
            source = Path(directory) / "source.png"
            target = Path(directory) / "result.gia"
            source.write_bytes(b"test")
            args = server._create_arg_parser().parse_args([
                "--cli", "--input", str(source), "--output", str(target),
                "--export", "container-node", "--name", "CLI容器", "--origin-x", "10", "--origin-y", "15",
                "--gia-mode", "classic",  # 素材组模式不改变客户端容器协议。
            ])
            with mock.patch.object(server.shaper_core, "process_image", return_value=sample_result()), \
                 mock.patch.object(server, "_load_json_to_gia", side_effect=AssertionError("must not use pyc")), \
                 mock.patch.object(server, "_load_convert_to_classic", side_effect=AssertionError("must not change mode")):
                self.assertEqual(server._run_cli(args), 0)
            root, main, units = decode(target.read_bytes())
            self.assertNotIn(4, root)
            self.assertEqual(wire._get(main, 3, 2).decode(), "CLI容器")
            self.assertEqual(wire._vector(layouts(units[1])[1][0][1], 504), [90, -65])
            args.output = None
            with mock.patch.object(server.shaper_core, "process_image", return_value=sample_result()), \
                 mock.patch.object(server.os, "getcwd", return_value=directory):
                self.assertEqual(server._run_cli(args), 0)
            self.assertTrue((Path(directory) / "CLI容器_container_node.gia").exists())

    @unittest.skipUnless(os.environ.get("UGC_GIA_CODEC"), "set UGC_GIA_CODEC to ugc-tool vendor dist/index.js")
    def test_independent_ugc_codec_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("OPENCODE_TEST_TMP")) as directory:
            fixtures = []
            for masked in (False, True):
                source = sample_result(mask=masked)
                path = Path(directory) / f"sample-{masked}.gia"
                path.write_bytes(client_ui_export.build_container_node_gia(source, "独立验证", origin_x=10, origin_y=15))
                path.with_suffix(".json").write_text(json.dumps(source), encoding="utf-8")
                fixtures.append(str(path))
            subprocess.run(["node", str(Path(__file__).with_suffix(".mjs")), os.environ["UGC_GIA_CODEC"], *fixtures], check=True)


if __name__ == "__main__":
    unittest.main()
