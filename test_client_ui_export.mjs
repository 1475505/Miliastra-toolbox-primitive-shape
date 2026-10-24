// Optional independent codec check; run by test_client_ui_export.py with UGC_GIA_CODEC set.
import assert from "node:assert/strict";
import fs from "node:fs";
import { pathToFileURL } from "node:url";

const { decode, encode } = await import(pathToFileURL(process.argv[2]).href);
const list = value => value === undefined ? [] : Array.isArray(value) ? value : [value];
const meta = unit => unit["19"]["1"];
const component = (unit, tag) => list(meta(unit)["505"]).find(part => String(tag) in part)?.["503"]?.[String(tag + 1)];
const platforms = unit => list(list(meta(unit)["505"]).find(part => "11" in part)?.["503"]?.["13"]?.["12"]?.["501"]);
// Like ugc-tool's prepareEncoding: an ambiguous decoded text can be represented
// as base64 even after dtype has converged to string across sibling messages.
function normalizeStrings(document) {
  const types = new Map(document.dtype_csv.split(/\r?\n/).map(line => {
    const [path, , type] = line.split(","); return [path, type];
  }));
  function visit(owner, path = "") {
    for (const [key, value] of Object.entries(owner)) {
      const field = path ? `${path}/${key}` : key;
      const convert = item => {
        if (types.get(field) === "string" && typeof item === "string" && item.startsWith("base64:")) {
          return `string:${Buffer.from(item.slice(7), "base64").toString("utf8")}`;
        }
        if (item && typeof item === "object") visit(item, field);
        return item;
      };
      owner[key] = Array.isArray(value) ? value.map(convert) : convert(value);
    }
  }
  visit(document.json);
  return document;
}

for (const filename of process.argv.slice(3)) {
  const document = decode(fs.readFileSync(filename), { type: "gia" });
  const source = JSON.parse(fs.readFileSync(filename.replace(/\.gia$/, ".json"), "utf8"));
  const masked = source.mask.enabled, offset = masked ? 2 : 1;
  const primary = document.json["1"], units = [primary, ...list(document.json["2"])];
  assert.equal(primary["5"], 70, "root must be a client control template, not a server UI group");
  assert.equal(units.length, source.elements.length + offset);
  assert.equal(component(primary, 72), undefined, "no server root-child reference wrapper");
  assert.ok(component(units[0], 78));
  const indices = [];
  for (const unit of units) {
    const id = unit["1"]["4"];
    assert.equal(unit["1"]["2"], 1); assert.equal(unit["1"]["3"], 8);
    assert.equal(meta(unit)["501"], id);
    assert.equal(list(meta(unit)["502"])[0]["11"]["501"], id);
    indices.push(list(meta(unit)["502"])[1]["12"]["501"]);
    assert.equal(unit["5"], unit === primary ? 70 : 15);
    for (const part of list(meta(unit)["505"])) {
      if (part["503"]) assert.equal(part["503"]["504"]["4"], id);
    }
    const devices = platforms(unit);
    assert.deepEqual(devices.map(p => p["501"] ?? 0), [0, 1, 2, 3]);
    for (const p of devices) assert.deepEqual(p["502"], devices[0]["502"]);
  }
  assert.equal(new Set(indices).size, units.length, "fresh child assets have independent indices");
  const rootLayout = platforms(primary)[0]["502"];
  assert.deepEqual(rootLayout["502"], { "501": .5, "502": .5 });
  assert.deepEqual(rootLayout["503"], { "501": .5, "502": .5 });
  assert.deepEqual(rootLayout["505"], { "501": source.image_size.width, "502": source.image_size.height });
  if (masked) {
    assert.equal(component(units[1], 83)["502"], 0x00ffffff);
    assert.equal(component(units[1], 84)["501"], 1);
  }
  const parent = masked ? units[1] : units[0];
  const images = units.slice(offset);
  assert.deepEqual(list(parent["2"]).map(ref => ref["4"]), images.map(unit => unit["1"]["4"]));
  assert.deepEqual(images.map(unit => component(unit, 83)["503"]), [100001, 100002, 100003]);
  assert.deepEqual(images.map(unit => component(unit, 83)["502"] >>> 0), [0xffffffff, 0x80123456, 0x00abcdef]);
  for (const [index, unit] of images.entries()) {
    assert.equal(meta(unit)["504"], parent["1"]["4"]);
    assert.equal(component(unit, 83)["504"], 1);
    assert.equal(component(unit, 84)["501"] ?? 0, 0);
    assert.equal(component(unit, 84)["509"] ?? 0, 0);
    const rect = platforms(unit)[0]["502"];
    const expected = source.elements[index];
    const px = masked ? source.mask.center.x * 2 : 10, py = masked ? source.mask.center.y * 2 : -15;
    assert.equal(rect["504"]["501"] ?? 0, expected.center.x * 2 - px);
    assert.equal(rect["504"]["502"] ?? 0, expected.center.y * 2 - py);
  }
  const ellipse = platforms(images[1])[0]["502"], triangle = platforms(images[2])[0]["502"];
  assert.deepEqual(ellipse["508"], { "1": 5, "2": 90, "3": 30 });
  assert.equal(triangle["508"]["3"], 45);
  assert.ok(Math.abs(triangle["506"]["502"] - 1 / 3) < 1e-6);
  normalizeStrings(document);
  assert.deepEqual(normalizeStrings(decode(encode(document, { type: "gia" }), { type: "gia" })).json, document.json);
  // Codec may canonicalize message field order once; subsequent binary output must be stable.
  const canonical = encode(document, { type: "gia" });
  assert.deepEqual(encode(normalizeStrings(decode(canonical, { type: "gia" })), { type: "gia" }), canonical);
  console.log(`PASS independent ugc-tool codec: client container-node template(70), mask=${masked}, asset indices, four layouts, geometry, ARGB and stable round-trip`);
}
