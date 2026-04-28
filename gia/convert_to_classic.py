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


def skip_proto_value(data, offset, wire_type):
    if wire_type == 0:
        _, offset = read_varint(data, offset)
        return offset
    if wire_type == 1:
        return offset + 8
    if wire_type == 2:
        length, offset = read_varint(data, offset)
        return offset + length
    if wire_type == 5:
        return offset + 4
    raise ValueError(f"Unsupported wire type {wire_type}")


def convert_gia_bytes_to_classic(data):
    if len(data) < 20:
        raise ValueError("File too small")

    content_len = int.from_bytes(data[16:20], "big")
    if content_len < 0 or len(data) < content_len + 20:
        raise ValueError("Invalid GIA content length")

    content = bytearray(data[20 : 20 + content_len])
    tail_magic = data[20 + content_len :]

    offset = 0
    insert_pos = len(content)
    remove_ranges = []

    while offset < len(content):
        start_pos = offset
        try:
            key, offset = read_varint(content, offset)
        except IndexError:
            break

        field_id = key >> 3
        wire_type = key & 0x07

        if field_id == 5:
            insert_pos = start_pos

        try:
            end_pos = skip_proto_value(content, offset, wire_type)
        except (IndexError, ValueError):
            break

        if field_id == 4:
            remove_ranges.append((start_pos, end_pos))
            if start_pos < insert_pos:
                insert_pos = start_pos

        offset = end_pos

    for start, end in reversed(remove_ranges):
        del content[start:end]

    removed_before_insert = sum(end - start for start, end in remove_ranges if start < insert_pos)
    insert_pos = max(0, insert_pos - removed_before_insert)
    content[insert_pos:insert_pos] = b"\x20\x01"

    new_content_len = len(content)
    new_file_size_field = new_content_len + 20

    new_data = bytearray()
    new_data.extend(new_file_size_field.to_bytes(4, "big"))
    new_data.extend(data[4:16])
    new_data.extend(new_content_len.to_bytes(4, "big"))
    new_data.extend(content)
    new_data.extend(tail_magic)
    return bytes(new_data)


def convert_to_classic(input_file, output_file):
    print(f"Converting {input_file} -> {output_file}")

    with open(input_file, "rb") as f:
        data = f.read()

    new_data = convert_gia_bytes_to_classic(data)
    with open(output_file, "wb") as out:
        out.write(new_data)

    print("Conversion successful.")

if __name__ == "__main__":
    if len(os.sys.argv) >= 3:
        convert_to_classic(os.sys.argv[1], os.sys.argv[2])
        raise SystemExit(0)

    convert_to_classic("gia/123冒险币.gia", "gia/123冒险币_converted.gia")
    
    # Verification
    if os.path.exists("gia/123冒险币_经典.gia"):
        print("-" * 40)
        print("Verifying against reference file...")
        with open("gia/123冒险币_converted.gia", "rb") as f1, open("gia/123冒险币_经典.gia", "rb") as f2:
            d1 = f1.read()
            d2 = f2.read()
            if d1 == d2:
                print("SUCCESS: Converted file is identical to reference classic file.")
            else:
                print(f"FAILURE: Files differ. Length: {len(d1)} vs {len(d2)}")
                # Simple diff print
                for i in range(min(len(d1), len(d2))):
                    if d1[i] != d2[i]:
                        print(f"First diff at offset {i}: {d1[i]:02X} vs {d2[i]:02X}")
                        break
