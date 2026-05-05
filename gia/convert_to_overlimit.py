import os


def read_varint(data, offset):
    result = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise IndexError("End of data while reading varint")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7f) << shift
        if not (byte & 0x80):
            return result, offset
        shift += 7


def convert_gia_bytes_to_overlimit(data):
    if len(data) < 20:
        raise ValueError("File too small")

    content_len = int.from_bytes(data[16:20], "big")
    if len(data) < content_len + 20:
        raise ValueError("Invalid GIA content length")

    content = bytearray(data[20 : 20 + content_len])
    tail_magic = data[20 + content_len :]

    offset = 0
    remove_ranges = []

    while offset < len(content):
        start_pos = offset
        try:
            key, offset = read_varint(content, offset)
        except IndexError:
            break

        field_id = key >> 3
        wire_type = key & 0x07

        if field_id == 4 and wire_type == 0:
            _, end_pos = read_varint(content, offset)
            remove_ranges.append((start_pos, end_pos))
            offset = end_pos
            continue

        if wire_type == 0:
            _, offset = read_varint(content, offset)
        elif wire_type == 1:
            offset += 8
        elif wire_type == 2:
            length, offset = read_varint(content, offset)
            offset += length
        elif wire_type == 5:
            offset += 4
        else:
            raise ValueError(f"Unknown wire type {wire_type} at offset {offset}")

    if not remove_ranges:
        # Field 4 not found, return original data unmodified
        return bytes(data)

    for start, end in reversed(remove_ranges):
        del content[start:end]

    new_content_len = len(content)
    new_file_size_field = new_content_len + 20

    new_data = bytearray()
    new_data.extend(new_file_size_field.to_bytes(4, "big"))
    new_data.extend(data[4:16])
    new_data.extend(new_content_len.to_bytes(4, "big"))
    new_data.extend(content)
    new_data.extend(tail_magic)
    return bytes(new_data)


def convert_to_overlimit(input_file, output_file):
    print(f"Converting {input_file} -> {output_file}")

    with open(input_file, "rb") as f:
        data = f.read()

    new_data = convert_gia_bytes_to_overlimit(data)
    with open(output_file, "wb") as out:
        out.write(new_data)

    print("Conversion successful.")


if __name__ == "__main__":
    if len(os.sys.argv) < 3:
        print("Usage: python gia/convert_to_overlimit.py <input.gia> <output.gia>")
        raise SystemExit(1)
    convert_to_overlimit(os.sys.argv[1], os.sys.argv[2])
