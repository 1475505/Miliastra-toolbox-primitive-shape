import unittest
from unittest import mock
import os
import sys

import cv2
import numpy as np

import fill_shaper
import primitive_backend
import server
import shaper_core

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gia"))
import json_to_gia
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "win"))
import app_desktop


class FillPipelineTests(unittest.TestCase):
    @staticmethod
    def _encode_png(image):
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise AssertionError("failed to encode test png")
        return buf.tobytes()

    def test_export_alpha_is_zeroed_only_when_png_alpha_weighting_is_enabled(self):
        shape = fill_shaper.Circle(cx=3.0, cy=3.0, radius_x=2.0, radius_y=2.0)
        alpha_weights = np.zeros((8, 8), dtype=np.float64)

        alpha_with_png_mode = 0.65
        if fill_shaper._shape_opacity(shape, alpha_weights, width=8, height=8) <= 0.05:
            alpha_with_png_mode = 0.0
        alpha_without_png_mode = 0.65

        self.assertEqual(alpha_with_png_mode, 0.0)
        self.assertAlmostEqual(alpha_without_png_mode, 0.65)

    def test_png_input_uses_white_flattening_when_png_mode_is_disabled(self):
        image = np.array(
            [[[0, 0, 255, 128], [255, 0, 0, 255]]],
            dtype=np.uint8,
        )

        fit_image, mask = shaper_core._extract_fill_image_and_mask(image, mask_threshold=127)

        self.assertEqual(fit_image.shape, (1, 2, 3))
        self.assertTrue(np.array_equal(mask, np.array([[True, True]])))
        self.assertTrue(np.array_equal(fit_image[0, 0], np.array([127, 127, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(fit_image[0, 1], np.array([255, 0, 0], dtype=np.uint8)))

    def test_primitive_png_target_whitens_hidden_rgb_and_compresses_soft_alpha(self):
        image = np.array(
            [[[0, 0, 255, 0], [0, 0, 255, 128], [255, 0, 0, 255]]],
            dtype=np.uint8,
        )

        target_image, mask, image_rgba = primitive_backend._extract_image_and_mask(
            image,
            mask_threshold=127,
            use_alpha_target=True,
        )

        self.assertEqual(target_image.shape, (1, 3, 4))
        self.assertTrue(np.array_equal(target_image[0, 0, :3], np.array([255, 255, 255], dtype=np.uint8)))
        self.assertEqual(int(target_image[0, 0, 3]), 0)
        self.assertTrue(np.array_equal(target_image[0, 1, :3], np.array([127, 127, 255], dtype=np.uint8)))
        self.assertLess(int(target_image[0, 1, 3]), 128)
        self.assertEqual(int(target_image[0, 2, 3]), 255)
        self.assertEqual(mask.shape, (1, 3))
        self.assertTrue(bool(mask[0, 1]))
        self.assertTrue(bool(mask[0, 2]))
        self.assertTrue(np.array_equal(image_rgba, image))

    def test_process_image_fill_adds_white_background_element_when_png_mode_disabled(self):
        image = np.array(
            [[[0, 0, 255, 0], [0, 255, 0, 255]]],
            dtype=np.uint8,
        )

        fitted_elements = [{"type": "ellipse", "center": {"x": 1.0, "y": -1.0}, "size": {"rx": 0.5, "ry": 0.5}}]
        with mock.patch.object(shaper_core.primitive_backend, "fit_image_with_primitive", return_value={"results": [], "preview": np.zeros((1, 2, 4), dtype=np.uint8)}), \
            mock.patch.object(shaper_core.fill_shaper, "results_to_elements", return_value=fitted_elements):
            result = shaper_core.process_image_fill(
                self._encode_png(image),
                {"enable_png_mode": False, "num_primitives": 1, "source_ext": ".png"},
            )

        self.assertFalse(result["config"]["output_has_transparency"])
        self.assertEqual(result["config"]["fill_variant"], "mask")
        self.assertTrue(result["config"]["source_is_png"])
        self.assertEqual(len(result["elements"]), 2)
        self.assertTrue(result["elements"][0]["is_background"])
        self.assertFalse(result["elements"][-1].get("is_background", False))
        self.assertEqual(result["elements"][0]["color"], "#ffffff")
        self.assertEqual(result["elements"][0]["center"], {"x": 1.0, "y": -0.5})
        self.assertEqual(result["elements"][0]["relative"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["elements"][0]["size"], {"width": 10.0, "height": 9.0})

    def test_process_image_fill_keeps_transparent_output_without_background_in_png_mode(self):
        image = np.array(
            [[[0, 0, 255, 0], [0, 255, 0, 255]]],
            dtype=np.uint8,
        )

        with mock.patch.object(shaper_core.primitive_backend, "fit_image_with_primitive", return_value={"results": [], "preview": np.zeros((1, 2, 4), dtype=np.uint8)}), \
            mock.patch.object(shaper_core.fill_shaper, "results_to_elements", return_value=[]):
            result = shaper_core.process_image_fill(
                self._encode_png(image),
                {"enable_png_mode": True, "num_primitives": 1, "source_ext": ".png"},
            )

        self.assertTrue(result["config"]["output_has_transparency"])
        self.assertEqual(result["config"]["fill_variant"], "png")
        self.assertEqual(result["elements"], [])

    def test_process_image_fill_does_not_add_background_for_opaque_inputs(self):
        image = np.array(
            [[[0, 0, 255], [0, 255, 0]]],
            dtype=np.uint8,
        )

        with mock.patch.object(shaper_core.primitive_backend, "fit_image_with_primitive", return_value={"results": [], "preview": np.zeros((1, 2, 4), dtype=np.uint8)}), \
            mock.patch.object(shaper_core.fill_shaper, "results_to_elements", return_value=[]):
            result = shaper_core.process_image_fill(
                self._encode_png(image),
                {"enable_png_mode": False, "num_primitives": 1, "source_ext": ".jpg"},
            )

        self.assertFalse(result["config"]["output_has_transparency"])
        self.assertEqual(result["elements"], [])

    def test_process_image_fill_does_not_add_background_for_transparent_non_png_inputs(self):
        image = np.array(
            [[[0, 0, 255, 0], [0, 255, 0, 255]]],
            dtype=np.uint8,
        )

        with mock.patch.object(shaper_core.primitive_backend, "fit_image_with_primitive", return_value={"results": [], "preview": np.zeros((1, 2, 4), dtype=np.uint8)}), \
             mock.patch.object(shaper_core.fill_shaper, "results_to_elements", return_value=[]):
            result = shaper_core.process_image_fill(
                self._encode_png(image),
                {"enable_png_mode": False, "num_primitives": 1, "source_ext": ".webp"},
            )

        self.assertFalse(result["config"]["source_is_png"])
        self.assertEqual(result["elements"], [])

    def test_png_mode_gia_export_keeps_mask_component_but_disables_it(self):
        tid = "test-png-mask-disabled"
        server.tasks[tid] = {
            "image_name": "bot.png",
            "result": {
                "image_center": {"x": 1.0, "y": 1.0},
                "config": {"pixel_per_unit": 1.0},
                "elements": [],
                "mask": {
                    "enabled": False,
                    "shape_type": "rectangle",
                    "center": {"x": 3.0, "y": -4.0},
                    "size": {"width": 8.0, "height": 6.0},
                },
            },
        }

        captured = {}

        class FakeGiaModule:
            MODE_IMAGE = "image"

            @staticmethod
            def convert_json_to_gia_bytes(json_data, base_gia_path, mode):
                captured["json_data"] = json_data
                return b"gia"

        try:
            with mock.patch.object(server, "_load_json_to_gia", return_value=FakeGiaModule()):
                with server.app.test_request_context(f"/download_overlimit_gia/{tid}"):
                    response = server.download_overlimit_gia(tid)
            self.assertEqual(response.status_code, 200)
            self.assertIsNotNone(captured["json_data"]["mask"])
            self.assertFalse(captured["json_data"]["mask"]["enabled"])
        finally:
            server.tasks.pop(tid, None)

    def test_gia_export_uses_custom_export_name_for_group_and_download(self):
        tid = "test-custom-gia-name"
        server.tasks[tid] = {
            "image_name": "original.png",
            "config": {},
            "result": {
                "image_center": {"x": 0.0, "y": 0.0},
                "config": {"pixel_per_unit": 1.0},
                "elements": [],
            },
        }

        captured = {}

        class FakeGiaModule:
            MODE_IMAGE = "image"

            @staticmethod
            def convert_json_to_gia_bytes(json_data, base_gia_path, mode):
                captured["json_data"] = json_data
                return b"gia"

        try:
            with mock.patch.object(server, "_load_json_to_gia", return_value=FakeGiaModule()):
                with server.app.test_request_context(f"/download_overlimit_gia/{tid}?export_name=renamed_asset"):
                    response = server.download_overlimit_gia(tid)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(captured["json_data"]["group_name"], "renamed_asset")
            self.assertIn("renamed_asset.gia", response.headers["Content-Disposition"])
        finally:
            server.tasks.pop(tid, None)

    def test_primitive_png_alpha_is_scaled_by_source_alpha(self):
        alpha_weights = np.zeros((16, 16), dtype=np.float64)
        alpha_weights[4:12, 4:12] = 0.4
        results = [{
            "type": "circle",
            "cx": 8.0,
            "cy": 8.0,
            "rx": 4.0,
            "ry": 4.0,
            "angle": 0.0,
            "color": "#ffffff",
            "alpha": 0.502,
            "packed_color": 0,
        }]

        weighted = primitive_backend._apply_alpha_weights_to_results(results, alpha_weights, 16, 16)

        self.assertLess(weighted[0]["alpha"], 0.502)
        self.assertGreater(weighted[0]["alpha"], 0.0)

    def test_export_basename_uses_uploaded_image_name(self):
        self.assertEqual(server._export_basename("demo_asset"), "demo_asset")
        self.assertEqual(server._export_basename("demo_asset.png"), "demo_asset")
        self.assertEqual(server._export_basename(""), "shaper_result")

    def test_healthz_returns_ok_payload(self):
        with server.app.test_client() as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True, "service": "primitive-shape"})

    def test_upload_page_uses_current_asset_version(self):
        with server.app.test_client() as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn(f"/web/style.css?v={server.WEB_ASSET_VERSION}", body)
        self.assertIn(f"/web/upload.js?v={server.WEB_ASSET_VERSION}", body)

    def test_desktop_save_helpers_support_css_exports(self):
        self.assertEqual(
            app_desktop._dialog_file_types("demo.css"),
            ("CSS files (*.css)", "All files (*.*)"),
        )
        self.assertTrue(
            app_desktop._resolve_save_path(r"C:\temp\demo", "demo.css").endswith(".css")
        )

    def test_result_page_exposes_copy_buttons_for_json_and_css(self):
        tid = "test-copy-buttons"
        server.tasks[tid] = {
            "image_name": "demo.png",
            "config": {"mode": "fill", "num_primitives": 40, "image_scale": 1.0, "output_alpha": 1.0},
            "result": {
                "mode": "fill",
                "elements_count": 0,
                "elapsed_seconds": 0.01,
                "image_size": {"width": 8, "height": 8},
                "image_center": {"x": 4, "y": 4},
                "elements": [],
                "config": {"pixel_per_unit": 1.0},
            },
        }
        try:
            with server.app.test_client() as client:
                response = client.get(f"/result/{tid}")
            self.assertEqual(response.status_code, 200)
            body = response.get_data(as_text=True)
            self.assertIn('id="btnCopyJSON"', body)
            self.assertIn('id="btnCopyCSS"', body)
        finally:
            server.tasks.pop(tid, None)

    def test_image_mode_export_order_keeps_background_first(self):
        ordered = json_to_gia._storage_order_elements_for_image_mode([
            {"type": "ellipse", "id": 1},
            {"type": "rectangle", "id": 99, "is_background": True},
            {"type": "triangle", "id": 2},
        ])
        self.assertEqual([item["id"] for item in ordered], [99, 2, 1])

    def test_image_mode_runtime_order_keeps_background_first(self):
        ordered = json_to_gia._order_elements_for_image_mode([
            {"type": "ellipse", "id": 1},
            {"type": "rectangle", "id": 99, "is_background": True},
            {"type": "triangle", "id": 2},
        ])
        self.assertEqual([item["id"] for item in ordered], [99, 2, 1])

    def test_triangle_results_export_preserves_base_and_height(self):
        results = [{
            "type": fill_shaper.ShapeType.TRIANGLE,
            "cx": 12.0,
            "cy": 8.0,
            "width": 6.0,
            "height": 10.0,
            "angle": 15.0,
            "color": "#abcdef",
            "alpha": 0.8,
        }]

        elements = fill_shaper.results_to_elements(results, unit_scale=0.5, img_center=(0.0, 0.0))

        self.assertEqual(elements[0]["type"], "triangle")
        self.assertEqual(elements[0]["size"], {"width": 3.0, "height": 5.0})

    def test_default_image_asset_refs_follow_shape_type(self):
        results = [
            {
                "type": fill_shaper.ShapeType.RECT,
                "cx": 10.0,
                "cy": 10.0,
                "hw": 2.0,
                "hh": 3.0,
                "angle": 0.0,
                "color": "#ffffff",
                "alpha": 1.0,
            },
            {
                "type": fill_shaper.ShapeType.TRIANGLE,
                "cx": 20.0,
                "cy": 20.0,
                "width": 6.0,
                "height": 8.0,
                "angle": 0.0,
                "color": "#ffffff",
                "alpha": 1.0,
            },
        ]

        elements = fill_shaper.results_to_elements(results, unit_scale=1.0, img_center=(0.0, 0.0))

        self.assertEqual(elements[0]["image_asset_ref"], 100001)
        self.assertEqual(elements[1]["image_asset_ref"], 100003)

    def test_json_to_gia_image_mode_uses_shape_specific_default_asset_refs(self):
        json_data = {
            "elements": [
                {
                    "type": "rectangle",
                    "relative": {"x": 1.0, "y": 2.0},
                    "size": {"width": 3.0, "height": 4.0},
                    "rotation": {"z": 0.0},
                },
                {
                    "type": "triangle",
                    "relative": {"x": 5.0, "y": 6.0},
                    "size": {"width": 7.0, "height": 8.0},
                    "rotation": {"z": 0.0},
                },
            ],
            "group_name": "shape-default-assets",
        }

        gia_bytes = json_to_gia.convert_json_to_gia_bytes(
            json_data=json_data,
            base_gia_path=os.path.join(os.path.dirname(__file__), "gia", "image_template.gia"),
            mode=json_to_gia.MODE_IMAGE,
        )

        _, _, root_fields, _ = json_to_gia._parse_gia_root_fields(gia_bytes)
        asset_refs = []
        for tag, wire, val in root_fields:
            if tag != 2 or not isinstance(val, bytes):
                continue
            info = json_to_gia.parse_resource_entry(val)
            if info["class"] != 15:
                continue

            entry_fields = json_to_gia.parse_message_fields(val)
            for entry_field in entry_fields:
                if entry_field["tag"] != 19 or entry_field["wire"] != json_to_gia.WireType.LENGTH_DELIMITED:
                    continue
                ui_fields = json_to_gia.parse_message_fields(entry_field["data"])
                for ui_field in ui_fields:
                    if ui_field["tag"] != 1 or ui_field["wire"] != json_to_gia.WireType.LENGTH_DELIMITED:
                        continue
                    content_fields = json_to_gia.parse_message_fields(ui_field["data"])
                    for content_field in content_fields:
                        if content_field["tag"] != 505 or content_field["wire"] != json_to_gia.WireType.LENGTH_DELIMITED:
                            continue
                        data_fields = json_to_gia.parse_message_fields(content_field["data"])
                        field_502 = next(
                            (
                                field["value"]
                                for field in data_fields
                                if field["tag"] == 502 and field["wire"] == json_to_gia.WireType.VARINT
                            ),
                            None,
                        )
                        if field_502 != 38:
                            continue
                        details = next(
                            (
                                field["data"]
                                for field in data_fields
                                if field["tag"] == 503 and field["wire"] == json_to_gia.WireType.LENGTH_DELIMITED
                            ),
                            None,
                        )
                        self.assertIsNotNone(details)
                        detail_fields = json_to_gia.parse_message_fields(details)
                        image_settings = next(
                            (
                                field["data"]
                                for field in detail_fields
                                if field["tag"] == 31 and field["wire"] == json_to_gia.WireType.LENGTH_DELIMITED
                            ),
                            None,
                        )
                        self.assertIsNotNone(image_settings)
                        image_settings_fields = json_to_gia.parse_message_fields(image_settings)
                        asset_ref = next(
                            (
                                field["value"]
                                for field in image_settings_fields
                                if field["tag"] == 2 and field["wire"] == json_to_gia.WireType.VARINT
                            ),
                            None,
                        )
                        if asset_ref is not None:
                            asset_refs.append(asset_ref)

        self.assertEqual(sorted(asset_refs), [100001, 100003])

    def test_json_to_gia_image_mode_exports_numeric_names(self):
        json_data = {
            "elements": [
                {
                    "type": "rectangle",
                    "relative": {"x": 1.0, "y": 2.0},
                    "size": {"width": 3.0, "height": 4.0},
                    "rotation": {"z": 0.0},
                },
                {
                    "type": "triangle",
                    "relative": {"x": 5.0, "y": 6.0},
                    "size": {"width": 7.0, "height": 8.0},
                    "rotation": {"z": 0.0},
                },
                {
                    "type": "ellipse",
                    "relative": {"x": 9.0, "y": 10.0},
                    "size": {"rx": 2.0, "ry": 3.0},
                    "rotation": {"z": 0.0},
                },
            ],
            "group_name": "numeric-names",
        }

        gia_bytes = json_to_gia.convert_json_to_gia_bytes(
            json_data=json_data,
            base_gia_path=os.path.join(os.path.dirname(__file__), "gia", "image_template.gia"),
            mode=json_to_gia.MODE_IMAGE,
        )

        _, _, root_fields, _ = json_to_gia._parse_gia_root_fields(gia_bytes)
        exported_names = []
        for tag, wire, val in root_fields:
            if tag != 2 or not isinstance(val, bytes):
                continue
            info = json_to_gia.parse_resource_entry(val)
            if info["class"] != 15:
                continue

            entry_fields = json_to_gia.parse_message_fields(val)
            for entry_field in entry_fields:
                if entry_field["tag"] != 19 or entry_field["wire"] != json_to_gia.WireType.LENGTH_DELIMITED:
                    continue
                ui_fields = json_to_gia.parse_message_fields(entry_field["data"])
                for ui_field in ui_fields:
                    if ui_field["tag"] != 1 or ui_field["wire"] != json_to_gia.WireType.LENGTH_DELIMITED:
                        continue
                    content_fields = json_to_gia.parse_message_fields(ui_field["data"])
                    for content_field in content_fields:
                        if content_field["tag"] != 505 or content_field["wire"] != json_to_gia.WireType.LENGTH_DELIMITED:
                            continue
                        data_fields = json_to_gia.parse_message_fields(content_field["data"])
                        field_502 = next(
                            (
                                field["value"]
                                for field in data_fields
                                if field["tag"] == 502 and field["wire"] == json_to_gia.WireType.VARINT
                            ),
                            None,
                        )
                        if field_502 != 15:
                            continue
                        name_wrapper = next(
                            (
                                field["data"]
                                for field in data_fields
                                if field["tag"] == 12 and field["wire"] == json_to_gia.WireType.LENGTH_DELIMITED
                            ),
                            None,
                        )
                        self.assertIsNotNone(name_wrapper)
                        name_fields = json_to_gia.parse_message_fields(name_wrapper)
                        name_value = next(
                            (
                                field["data"].decode("utf-8")
                                for field in name_fields
                                if field["tag"] == 501 and field["wire"] == json_to_gia.WireType.LENGTH_DELIMITED
                            ),
                            None,
                        )
                        if name_value:
                            exported_names.append(name_value)

        self.assertEqual(sorted(exported_names, key=int), ["1", "2", "3"])

    def test_triangle_rasterize_uses_independent_base_and_height(self):
        tall = fill_shaper.Triangle(cx=20.0, cy=20.0, base_width=8.0, height=14.0)
        wide = fill_shaper.Triangle(cx=20.0, cy=20.0, base_width=14.0, height=8.0)

        tall_ys, tall_xs, _ = tall.rasterize(width=48, height=48)
        wide_ys, wide_xs, _ = wide.rasterize(width=48, height=48)

        self.assertGreater(len(tall_xs), 0)
        self.assertGreater(len(wide_xs), 0)
        self.assertLess(int(tall_xs.max() - tall_xs.min()), int(wide_xs.max() - wide_xs.min()))
        self.assertGreater(int(tall_ys.max() - tall_ys.min()), int(wide_ys.max() - wide_ys.min()))


if __name__ == "__main__":
    unittest.main()
