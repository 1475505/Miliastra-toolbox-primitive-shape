import io
import struct
import unittest

import gia_lua
import server


def vi(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    return bytes(out + bytes([value]))


def field(tag, value):
    if isinstance(value, bytes):
        return vi(tag*8+2) + vi(len(value)) + value
    if isinstance(value, float):
        return vi(tag*8+5) + struct.pack('<f', value)
    return vi(tag*8) + vi(value)


def vec(x, y, start=501):
    return field(start, float(x)) + field(start+1, float(y))


def fixture(*, packed=False, classic=False, mask=False, missing=False, image=True):
    children = [2, 3] + ([999] if missing else [])
    refs = field(503, b''.join(vi(x) for x in children)) if packed else b''.join(field(503,x) for x in children)
    root_content = field(501,1) + refs
    if mask:
        root_content += field(505,field(502,56)+field(503,field(47,field(4,1))))
    root = field(5,61) + field(19,field(1,root_content))
    data = field(1,root)
    for guid, asset, color in [(2,100003,0x80123456),(3,100001,0xffabcdef)]:
        transform = (field(501,vec(-2,3,1)+field(3,1.0)) + field(502,vec(.5,.5))
                     + field(503,vec(.5,.5)) + field(504,vec(guid*10,-20))
                     + field(505,vec(40,60)) + field(506,vec(.25,1/3)) + field(508,field(3,30.0)))
        platform = field(501,field(502,transform))
        component = field(502,12)+field(503,field(13,field(12,platform)))
        content = field(501,guid)+field(504,1)+field(505,component)
        if image:
            content += field(505,field(502,38)+field(503,field(31,field(2,asset)+field(4,color))))
        data += field(2,field(5,15)+field(19,field(1,content)))
    if classic:
        data += field(4,1)
    return struct.pack('>5I',len(data)+20,1,0x326,3,len(data))+data+struct.pack('>I',0x679)


class GiaLuaTests(unittest.TestCase):
    def test_geometry_color_pivot_mirror_and_painter_order(self):
        records = gia_lua.parse_material_gia(fixture())['records']
        self.assertEqual([r[0] for r in records],[100001,100003])
        row = records[1]
        self.assertEqual(row[1:6],[20,-20,40,60,.25])
        self.assertAlmostEqual(row[6],1/3,places=6)
        self.assertEqual(row[11:],[ -2,3,30,0x12,0x34,0x56,0x80])

    def test_packed_children_matches_unpacked(self):
        self.assertEqual(gia_lua.parse_material_gia(fixture(packed=True)),gia_lua.parse_material_gia(fixture()))

    def test_mask_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError,'遮罩'):
            gia_lua.build_gia_lua(fixture(mask=True))
        self.assertIn('忽略原素材组遮罩',gia_lua.build_gia_lua(fixture(mask=True),ignore_mask=True))

    def test_missing_legacy_references_are_reported(self):
        self.assertEqual(gia_lua.parse_material_gia(fixture(missing=True))['missing_refs'],[999])
        self.assertIn('1 个没有图片实体',gia_lua.build_gia_lua(fixture(missing=True)))

    def test_rejects_wrong_mode_invalid_header_and_nonimage(self):
        for blob in [b'',fixture()[:-1],b'xxxx'+fixture()[4:],fixture(classic=True),fixture(image=False)]:
            with self.subTest(size=len(blob)), self.assertRaises(ValueError):
                gia_lua.build_gia_lua(blob)

    def test_truncated_protobuf_and_varint(self):
        for data in [b'\x0a\xff',b'\x0a\x05a',b'\x00',b'\x08'+b'\xff'*10]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                gia_lua._fields(data)

    def test_download_route_and_tutorial(self):
        client = server.app.test_client()
        response = client.post('/convert_gia_mode',data={'direction':'gia_to_lua','gia':(io.BytesIO(fixture()),'drawing.gia')})
        self.assertEqual(response.status_code,200)
        self.assertIn('.lua',response.headers['Content-Disposition'])
        self.assertIn('text/plain',response.content_type)
        self.assertIn('image:SetImage(Enum.ImageSource.StaticReference, item[1])', response.text)
        self.assertNotIn('PREFAB_ID_BY_ASSET', response.text)
        for text in ['IMAGE_PREFAB_ID','控件模板索引ID','OnStart','BASE_SCALE','100003']:
            self.assertIn(text,response.text)
        response = client.post('/convert_gia_mode',data={'direction':'gia_to_lua','gia':(io.BytesIO(fixture(mask=True)),'drawing.gia')})
        self.assertEqual(response.status_code,400)
        response = client.post('/convert_gia_mode',data={'direction':'gia_to_lua','ignore_mask':'1','gia':(io.BytesIO(fixture(mask=True)),'drawing.gia')})
        self.assertEqual(response.status_code,200)

    def test_existing_mode_conversion_still_works(self):
        client = server.app.test_client()
        for direction in ['classic_to_overlimit','overlimit_to_classic']:
            result = client.post('/convert_gia_mode',data={'direction':direction,'gia':(io.BytesIO(fixture()),'drawing.gia')})
            self.assertEqual(result.status_code,200)
            self.assertIn('.gia',result.headers['Content-Disposition'])


if __name__ == '__main__':
    unittest.main()
