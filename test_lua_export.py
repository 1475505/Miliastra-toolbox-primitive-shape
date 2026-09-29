import unittest

import lua_export


def result(*rotations):
    elements = [{"type": "rectangle", "center": {"x": 10, "y": 20}, "size": {"width": 4, "height": 6},
                 "rotation": rotation, "color": "#123456", "alpha": 1.0} for rotation in rotations]
    return {"mode": "fill", "image_size": {"width": 100, "height": 80}, "config": {}, "elements": elements}


class LuaExportTests(unittest.TestCase):
    def test_xy_rotation_is_kept(self):
        text = lua_export.build_lua_export_text(result({"x": 5, "y": 90, "z": 30}))
        self.assertIn("{0, 10.0, 20.0, 4.0, 6.0, 30.0, 1, 255, 5.0, 90.0},", text)
        self.assertIn("image:SetLocalRotation(rotX, rotY, rotZ)", text)

    def test_zero_xy_keeps_eight_fields(self):
        text = lua_export.build_lua_export_text(result({"x": 0, "y": 0, "z": 30}, 45))
        self.assertIn("{0, 10.0, 20.0, 4.0, 6.0, 30.0, 1, 255},", text)
        self.assertIn("{0, 10.0, 20.0, 4.0, 6.0, 45.0, 1, 255},", text)


if __name__ == '__main__':
    unittest.main()
