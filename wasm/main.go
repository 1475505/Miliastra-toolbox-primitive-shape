//go:build js && wasm

// WebAssembly wrapper for the fogleman/primitive fitting engine.
//
// The browser feeds a pre-processed RGBA work image (white-flattened or
// alpha-compressed, mirroring primitive_backend.py) plus fitting options.
// This program runs the same Model/Step loop as the primitive CLI and
// returns a JSON document with the fitted shapes (full-resolution
// coordinates), a PNG preview, mask information and the alpha-weighted
// colors used by the transparent-PNG flow.
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"math"
	"syscall/js"
	"time"

	"github.com/fogleman/primitive/primitive"
	"github.com/nfnt/resize"
)

const (
	shapeModeTriangle = 1
	shapeModeRect     = 5
	shapeModeCircle   = 7
)

// keep in sync with primitive_backend.SHAPE_ORDER
var shapeOrder = []string{"circle", "rect", "triangle"}

var shapeModeMap = map[string]int{
	"circle":   shapeModeCircle,
	"rect":     shapeModeRect,
	"triangle": shapeModeTriangle,
}

type fitConfig struct {
	FullW         int      `json:"full_w"`
	FullH         int      `json:"full_h"`
	WorkW         int      `json:"work_w"`
	WorkH         int      `json:"work_h"`
	NumPrimitives int      `json:"num_primitives"`
	AllowedShapes []string `json:"allowed_shapes"`
	Transparent   bool     `json:"transparent"`
	MaskThreshold int      `json:"mask_threshold"`
}

type shapeJSON struct {
	Type        string  `json:"type"`
	CX          float64 `json:"cx"`
	CY          float64 `json:"cy"`
	RX          float64 `json:"rx,omitempty"`
	RY          float64 `json:"ry,omitempty"`
	HW          float64 `json:"hw,omitempty"`
	HH          float64 `json:"hh,omitempty"`
	Width       float64 `json:"width,omitempty"`
	Height      float64 `json:"height,omitempty"`
	Size        float64 `json:"size,omitempty"`
	Angle       float64 `json:"angle"`
	Color       string  `json:"color"`
	Alpha       float64 `json:"alpha"`
	PackedColor uint32  `json:"packed_color"`
}

type fitResult struct {
	Shapes     []shapeJSON `json:"shapes"`
	PreviewPNG string      `json:"preview_png"`
	MaskPNG    string      `json:"mask_png,omitempty"`
	BBox       [4]int      `json:"bbox"` // full-resolution x0,y0,x1,y1
	Coverage   float64     `json:"coverage"`
	Error      string      `json:"error,omitempty"`
}

// fitSession holds the per-instance fitting state for the distributed
// step-parallel pipeline (init → state/search → apply → finish).
type fitSession struct {
	cfg       fitConfig
	model     *primitive.Model
	rgba      []byte
	origAlpha []byte
	total     int
	done      int
	configs   [][2]int
}

var session *fitSession

// candidateJSON is the wire format of one searched shape candidate. The
// geometry stays in work-image coordinates; color is recomputed by
// Model.Add on the main instance, mirroring the CLI's runWorkers flow.
type candidateJSON struct {
	Type   string  `json:"type"`
	X      float64 `json:"x,omitempty"`
	Y      float64 `json:"y,omitempty"`
	Rx     float64 `json:"rx,omitempty"`
	Ry     float64 `json:"ry,omitempty"`
	Angle  float64 `json:"angle,omitempty"`
	Xi     int     `json:"xi,omitempty"`
	Yi     int     `json:"yi,omitempty"`
	Rxi    int     `json:"rxi,omitempty"`
	Ryi    int     `json:"ryi,omitempty"`
	Circle bool    `json:"circle,omitempty"`
	Sx     int     `json:"sx,omitempty"`
	Sy     int     `json:"sy,omitempty"`
	AngleI int     `json:"anglei,omitempty"`
	X1     int     `json:"x1,omitempty"`
	Y1     int     `json:"y1,omitempty"`
	X2     int     `json:"x2,omitempty"`
	Y2     int     `json:"y2,omitempty"`
	X3     int     `json:"x3,omitempty"`
	Y3     int     `json:"y3,omitempty"`
	Alpha  int     `json:"alpha"`
	Energy float64 `json:"energy"`
}

func normalizeShapes(requested []string) []string {
	seen := map[string]bool{}
	for _, name := range requested {
		seen[name] = true
	}
	normalized := []string{}
	for _, name := range shapeOrder {
		if seen[name] {
			normalized = append(normalized, name)
		}
	}
	for _, name := range requested {
		if _, ok := shapeModeMap[name]; ok && !contains(normalized, name) {
			normalized = append(normalized, name)
		}
	}
	if len(normalized) == 0 {
		normalized = []string{"circle"}
	}
	return normalized
}

func contains(list []string, v string) bool {
	for _, item := range list {
		if item == v {
			return true
		}
	}
	return false
}

// mirror primitive_backend._build_shape_configs
func buildShapeConfigs(allowed []string, total int) [][2]int {
	normalized := normalizeShapes(allowed)
	if total < 1 {
		total = 1
	}
	base := total / len(normalized)
	remainder := total % len(normalized)
	configs := [][2]int{}
	for i, name := range normalized {
		count := base
		if i < remainder {
			count++
		}
		if count <= 0 {
			continue
		}
		configs = append(configs, [2]int{shapeModeMap[name], count})
	}
	if len(configs) == 0 {
		configs = append(configs, [2]int{shapeModeCircle, total})
	}
	return configs
}

func packColor(r, g, b int, alpha float64) uint32 {
	a := int(math.Round(alpha * 255.0))
	if a < 0 {
		a = 0
	}
	if a > 255 {
		a = 255
	}
	return uint32(a)<<24 | uint32(r&0xff)<<16 | uint32(g&0xff)<<8 | uint32(b&0xff)
}

func hexColor(c primitive.Color) string {
	return string('#') + hex2(c.R) + hex2(c.G) + hex2(c.B)
}

func hex2(v int) string {
	const digits = "0123456789abcdef"
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return string([]byte{digits[(v>>4)&0xf], digits[v&0xf]})
}

// binaryMask builds the foreground mask on the work image.
// transparent flow: origAlpha > 0 (no cleanup, mirrors shaper_core).
// alpha flow: origAlpha >= threshold, then 3x3 close+open.
// opaque flow: border-sampled background color distance, then 3x3 close+open.
func binaryMask(rgba []byte, origAlpha []byte, w, h int, transparent bool, threshold int) []bool {
	mask := make([]bool, w*h)
	if transparent {
		for i := 0; i < w*h; i++ {
			mask[i] = origAlpha[i] > 0
		}
		return mask
	}
	hasTransparent := false
	for i := 0; i < w*h; i++ {
		if origAlpha[i] < 255 {
			hasTransparent = true
			break
		}
	}
	if hasTransparent {
		t := uint8(threshold)
		for i := 0; i < w*h; i++ {
			mask[i] = origAlpha[i] >= t
		}
	} else {
		bgR, bgG, bgB := borderColor(rgba, w, h)
		const distThresh = 40.0
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				i := y*w + x
				dr := float64(int(rgba[i*4]) - bgR)
				dg := float64(int(rgba[i*4+1]) - bgG)
				db := float64(int(rgba[i*4+2]) - bgB)
				mask[i] = math.Sqrt(dr*dr+dg*dg+db*db) > distThresh
			}
		}
	}
	return morphologyClean(mask, w, h)
}

// borderColor estimates the background color from the image border pixels.
func borderColor(rgba []byte, w, h int) (int, int, int) {
	var sr, sg, sb, n int
	sample := func(x, y int) {
		i := (y*w + x) * 4
		sr += int(rgba[i])
		sg += int(rgba[i+1])
		sb += int(rgba[i+2])
		n++
	}
	step := w / 64
	if step < 1 {
		step = 1
	}
	for x := 0; x < w; x += step {
		sample(x, 0)
		sample(x, h-1)
	}
	step = h / 64
	if step < 1 {
		step = 1
	}
	for y := 0; y < h; y += step {
		sample(0, y)
		sample(w-1, y)
	}
	if n == 0 {
		return 255, 255, 255
	}
	return sr / n, sg / n, sb / n
}

func dilate(mask []bool, w, h int) []bool {
	out := make([]bool, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			on := false
			for dy := -1; dy <= 1 && !on; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := x+dx, y+dy
					if nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny*w+nx] {
						on = true
						break
					}
				}
			}
			out[y*w+x] = on
		}
	}
	return out
}

func erode(mask []bool, w, h int) []bool {
	out := make([]bool, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			on := true
			for dy := -1; dy <= 1 && on; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := x+dx, y+dy
					if nx < 0 || nx >= w || ny < 0 || ny >= h || !mask[ny*w+nx] {
						on = false
						break
					}
				}
			}
			out[y*w+x] = on
		}
	}
	return out
}

// morphologyClean applies close (dilate+erode) then open (erode+dilate),
// mirroring the 3x3 ellipse cleanup in shaper_core.
func morphologyClean(mask []bool, w, h int) []bool {
	any, all := false, true
	for _, v := range mask {
		if v {
			any = true
		} else {
			all = false
		}
	}
	if !any {
		return mask
	}
	if all {
		return mask
	}
	closed := erode(dilate(mask, w, h), w, h)
	return dilate(erode(closed, w, h), w, h)
}

func maskPNG(mask []bool, w, h int) (string, error) {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for i, v := range mask {
		if v {
			img.Pix[i] = 255
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

func maskBBox(mask []bool, w, h int) [4]int {
	x0, y0, x1, y1 := w, h, -1, -1
	count := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mask[y*w+x] {
				if x < x0 {
					x0 = x
				}
				if x > x1 {
					x1 = x
				}
				if y < y0 {
					y0 = y
				}
				if y > y1 {
					y1 = y
				}
				count++
			}
		}
	}
	if count == 0 {
		return [4]int{0, 0, w, h}
	}
	return [4]int{x0, y0, x1 + 1, y1 + 1}
}

// shapeOpacity computes the mean alpha-map coverage of a shape's rasterized
// scanlines, mirroring fill_shaper._shape_opacity.
func shapeOpacity(shape primitive.Shape, alphaMap []byte, w, h int) float64 {
	lines := shape.Rasterize()
	var weighted, total float64
	for _, line := range lines {
		if line.Y < 0 || line.Y >= h {
			continue
		}
		a := float64(line.Alpha) / 65535.0
		for x := line.X1; x <= line.X2; x++ {
			if x < 0 || x >= w {
				continue
			}
			weighted += a * float64(alphaMap[line.Y*w+x]) / 255.0
			total += a
		}
	}
	if total <= 1e-8 {
		return 0
	}
	return weighted / total
}

func exportShapes(model *primitive.Model, sx, sy float64, alphaMap []byte, workW, workH int, transparent bool) []shapeJSON {
	results := []shapeJSON{}
	for i, shape := range model.Shapes {
		color := model.Colors[i]
		alpha := float64(color.A) / 255.0
		if transparent && alphaMap != nil {
			opacity := shapeOpacity(shape, alphaMap, workW, workH)
			if opacity <= 0.05 {
				alpha = 0
			} else {
				alpha *= opacity
			}
			if alpha < 0 {
				alpha = 0
			}
			if alpha > 1 {
				alpha = 1
			}
		}
		entry := shapeJSON{
			Angle:       0,
			Color:       hexColor(color),
			Alpha:       round4(alpha),
			PackedColor: packColor(color.R, color.G, color.B, alpha),
		}
		switch s := shape.(type) {
		case *primitive.RotatedEllipse:
			entry.Type = "circle"
			entry.CX = round4(s.X * sx)
			entry.CY = round4(s.Y * sy)
			entry.RX = round4(s.Rx * sx)
			entry.RY = round4(s.Ry * sy)
			entry.Angle = round4(s.Angle)
		case *primitive.Ellipse:
			entry.Type = "circle"
			entry.CX = round4(float64(s.X) * sx)
			entry.CY = round4(float64(s.Y) * sy)
			entry.RX = round4(float64(s.Rx) * sx)
			entry.RY = round4(float64(s.Ry) * sy)
		case *primitive.RotatedRectangle:
			entry.Type = "rect"
			entry.CX = round4(float64(s.X) * sx)
			entry.CY = round4(float64(s.Y) * sy)
			entry.HW = round4(math.Max(float64(s.Sx)*sx/2.0, 0.5))
			entry.HH = round4(math.Max(float64(s.Sy)*sy/2.0, 0.5))
			entry.Angle = round4(float64(s.Angle))
		case *primitive.Rectangle:
			x1, y1, x2, y2 := s.X1, s.Y1, s.X2, s.Y2
			if x1 > x2 {
				x1, x2 = x2, x1
			}
			if y1 > y2 {
				y1, y2 = y2, y1
			}
			entry.Type = "rect"
			entry.CX = round4(float64(x1+x2) / 2.0 * sx)
			entry.CY = round4(float64(y1+y2) / 2.0 * sy)
			entry.HW = round4(math.Max(float64(x2-x1)*sx/2.0, 0.5))
			entry.HH = round4(math.Max(float64(y2-y1)*sy/2.0, 0.5))
		case *primitive.Triangle:
			// mirror primitive_backend._parse_triangle
			px := []float64{float64(s.X1) * sx, float64(s.X2) * sx, float64(s.X3) * sx}
			py := []float64{float64(s.Y1) * sy, float64(s.Y2) * sy, float64(s.Y3) * sy}
			cx := (px[0] + px[1] + px[2]) / 3.0
			cy := (py[0] + py[1] + py[2]) / 3.0
			minX, maxX := math.Min(px[0], math.Min(px[1], px[2])), math.Max(px[0], math.Max(px[1], px[2]))
			minY, maxY := math.Min(py[0], math.Min(py[1], py[2])), math.Max(py[0], math.Max(py[1], py[2]))
			width := math.Max(maxX-minX, 1.0)
			height := math.Max(maxY-minY, 1.0)
			edgeX, edgeY := px[1]-px[0], py[1]-py[0]
			angle := math.Atan2(edgeY, edgeX)*180.0/math.Pi + 90.0
			entry.Type = "triangle"
			entry.CX = round4(cx)
			entry.CY = round4(cy)
			entry.Width = round4(width)
			entry.Height = round4(height)
			entry.Size = round4(math.Max(width, height))
			entry.Angle = round4(angle)
		default:
			continue
		}
		results = append(results, entry)
	}
	return results
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}

func normalizeCfg(cfg fitConfig) fitConfig {
	if cfg.NumPrimitives < 1 {
		cfg.NumPrimitives = 400
	}
	if cfg.MaskThreshold < 1 {
		cfg.MaskThreshold = 1
	}
	if cfg.MaskThreshold > 254 {
		cfg.MaskThreshold = 254
	}
	return cfg
}

// buildSession creates the primitive model shared by both the one-shot and
// the distributed step-parallel pipelines.
func buildSession(cfg fitConfig, rgba []byte, origAlpha []byte) *fitSession {
	cfg = normalizeCfg(cfg)
	img := image.NewRGBA(image.Rect(0, 0, cfg.WorkW, cfg.WorkH))
	copy(img.Pix, rgba)

	var bg primitive.Color
	if cfg.Transparent {
		bg = primitive.MakeHexColor("ffffff00")
	} else {
		bg = primitive.MakeHexColor("ffffff")
	}

	outputSize := cfg.WorkW
	if cfg.WorkH > outputSize {
		outputSize = cfg.WorkH
	}
	model := primitive.NewModel(img, bg, outputSize, 1)
	configs := buildShapeConfigs(cfg.AllowedShapes, cfg.NumPrimitives)
	total := 0
	for _, pair := range configs {
		total += pair[1]
	}
	return &fitSession{
		cfg:       cfg,
		model:     model,
		rgba:      rgba,
		origAlpha: origAlpha,
		total:     total,
		configs:   configs,
	}
}

// finalizeSession renders the mask/preview/exports for a finished session.
func finalizeSession(s *fitSession) fitResult {
	cfg := s.cfg
	sx := float64(cfg.FullW) / float64(cfg.WorkW)
	sy := float64(cfg.FullH) / float64(cfg.WorkH)

	// foreground mask on the work image
	mask := binaryMask(s.rgba, s.origAlpha, cfg.WorkW, cfg.WorkH, cfg.Transparent, cfg.MaskThreshold)
	bbox := maskBBox(mask, cfg.WorkW, cfg.WorkH)
	fullBBox := [4]int{
		int(math.Round(float64(bbox[0]) * sx)),
		int(math.Round(float64(bbox[1]) * sy)),
		int(math.Round(float64(bbox[2]) * sx)),
		int(math.Round(float64(bbox[3]) * sy)),
	}
	coverage := 0.0
	{
		on := 0
		for _, v := range mask {
			if v {
				on++
			}
		}
		coverage = float64(on) / float64(cfg.WorkW*cfg.WorkH)
	}

	// preview from the model canvas
	previewRGBA := imageToNRGBA(s.model.Context.Image())
	if cfg.Transparent && s.origAlpha != nil {
		sw, sh := previewRGBA.Rect.Dx(), previewRGBA.Rect.Dy()
		resized := resize.Resize(uint(sw), uint(sh), alphaImage(s.origAlpha, cfg.WorkW, cfg.WorkH), resize.Bilinear)
		na := resized.(*image.Gray)
		for y := 0; y < sh; y++ {
			for x := 0; x < sw; x++ {
				i := y*sw + x
				previewRGBA.Pix[i*4+3] = uint8(int(previewRGBA.Pix[i*4+3]) * int(na.Pix[i]) / 255)
			}
		}
	}
	var previewBuf bytes.Buffer
	if err := png.Encode(&previewBuf, previewRGBA); err != nil {
		return fitResult{Error: err.Error()}
	}

	result := fitResult{
		Shapes:     exportShapes(s.model, sx, sy, s.origAlpha, cfg.WorkW, cfg.WorkH, cfg.Transparent),
		PreviewPNG: base64.StdEncoding.EncodeToString(previewBuf.Bytes()),
		BBox:       fullBBox,
		Coverage:   round4(coverage),
	}
	if !cfg.Transparent {
		if maskB64, err := maskPNG(mask, cfg.WorkW, cfg.WorkH); err == nil {
			result.MaskPNG = maskB64
		}
	}
	return result
}

func runFit(cfg fitConfig, rgba []byte, origAlpha []byte) fitResult {
	s := buildSession(cfg, rgba, origAlpha)

	progressFn := js.Global().Get("primitiveFitProgress")
	for _, pair := range s.configs {
		mode, count := pair[0], pair[1]
		for i := 0; i < count; i++ {
			s.model.Step(primitive.ShapeType(mode), 0, 0)
			s.done++
			if progressFn.Type() == js.TypeFunction {
				progressFn.Invoke(s.done, s.total)
			}
		}
	}
	return finalizeSession(s)
}

// alphaImage wraps a uint8 alpha plane as *image.Gray.
func alphaImage(pix []byte, w, h int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	copy(img.Pix, pix)
	return img
}

// imageToNRGBA converts an image to *image.NRGBA.
func imageToNRGBA(src image.Image) *image.NRGBA {
	b := src.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			r, g, bl, a := src.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := (y*b.Dx() + x) * 4
			out.Pix[i] = uint8(r >> 8)
			out.Pix[i+1] = uint8(g >> 8)
			out.Pix[i+2] = uint8(bl >> 8)
			out.Pix[i+3] = uint8(a >> 8)
		}
	}
	return out
}

func fitFunc() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return `{"error":"missing arguments"}`
		}
		cfgJSON := args[0].String()
		rgbaArray := args[1]
		alphaArray := args[2]

		var cfg fitConfig
		if err := json.Unmarshal([]byte(cfgJSON), &cfg); err != nil {
			b, _ := json.Marshal(fitResult{Error: "config parse: " + err.Error()})
			return string(b)
		}
		if cfg.WorkW < 1 || cfg.WorkH < 1 || cfg.FullW < 1 || cfg.FullH < 1 {
			b, _ := json.Marshal(fitResult{Error: "invalid dimensions"})
			return string(b)
		}

		rgba := make([]byte, rgbaArray.Get("length").Int())
		js.CopyBytesToGo(rgba, rgbaArray)
		if len(rgba) != cfg.WorkW*cfg.WorkH*4 {
			b, _ := json.Marshal(fitResult{Error: "rgba size mismatch"})
			return string(b)
		}
		var alpha []byte
		if alphaArray.InstanceOf(js.Global().Get("Uint8Array")) && alphaArray.Get("length").Int() == cfg.WorkW*cfg.WorkH {
			alpha = make([]byte, cfg.WorkW*cfg.WorkH)
			js.CopyBytesToGo(alpha, alphaArray)
		} else {
			alpha = make([]byte, cfg.WorkW*cfg.WorkH)
			for i := range alpha {
				alpha[i] = 255
			}
		}

		result := runFit(cfg, rgba, alpha)
		b, err := json.Marshal(result)
		if err != nil {
			b, _ = json.Marshal(fitResult{Error: err.Error()})
		}
		return string(b)
	})
}

/* ---------------- distributed step-parallel API ----------------
 *
 * Mirrors the CLI's `primitive -j N` semantics: within each step, N searchers
 * (each an independent WASM instance) run BestHillClimbState from the same
 * canvas state; the coordinator picks the lowest-energy candidate and the
 * main instance applies it via Model.Add (which recomputes the color).
 */

func exportCandidate(state *primitive.State) candidateJSON {
	cj := candidateJSON{Alpha: state.Alpha, Energy: state.Energy()}
	switch s := state.Shape.(type) {
	case *primitive.RotatedEllipse:
		cj.Type = "circle"
		cj.X, cj.Y = s.X, s.Y
		cj.Rx, cj.Ry = s.Rx, s.Ry
		cj.Angle = s.Angle
	case *primitive.Ellipse:
		cj.Type = "ellipse"
		cj.Xi, cj.Yi = s.X, s.Y
		cj.Rxi, cj.Ryi = s.Rx, s.Ry
		cj.Circle = s.Circle
	case *primitive.RotatedRectangle:
		cj.Type = "rect"
		cj.Xi, cj.Yi = s.X, s.Y
		cj.Sx, cj.Sy = s.Sx, s.Sy
		cj.AngleI = s.Angle
	case *primitive.Rectangle:
		cj.Type = "rectAxis"
		cj.X1, cj.Y1 = s.X1, s.Y1
		cj.X2, cj.Y2 = s.X2, s.Y2
	case *primitive.Triangle:
		cj.Type = "triangle"
		cj.X1, cj.Y1 = s.X1, s.Y1
		cj.X2, cj.Y2 = s.X2, s.Y2
		cj.X3, cj.Y3 = s.X3, s.Y3
	}
	return cj
}

func buildShape(worker *primitive.Worker, cj candidateJSON) primitive.Shape {
	switch cj.Type {
	case "circle":
		return &primitive.RotatedEllipse{Worker: worker, X: cj.X, Y: cj.Y, Rx: cj.Rx, Ry: cj.Ry, Angle: cj.Angle}
	case "ellipse":
		return &primitive.Ellipse{Worker: worker, X: cj.Xi, Y: cj.Yi, Rx: cj.Rxi, Ry: cj.Ryi, Circle: cj.Circle}
	case "rect":
		return &primitive.RotatedRectangle{Worker: worker, X: cj.Xi, Y: cj.Yi, Sx: cj.Sx, Sy: cj.Sy, Angle: cj.AngleI}
	case "rectAxis":
		return &primitive.Rectangle{Worker: worker, X1: cj.X1, Y1: cj.Y1, X2: cj.X2, Y2: cj.Y2}
	case "triangle":
		return &primitive.Triangle{Worker: worker, X1: cj.X1, Y1: cj.Y1, X2: cj.X2, Y2: cj.Y2, X3: cj.X3, Y3: cj.Y3}
	}
	return nil
}

func marshalOrError(v interface{}, prefix string) string {
	b, err := json.Marshal(v)
	if err != nil {
		b, _ = json.Marshal(map[string]string{"error": prefix + err.Error()})
	}
	return string(b)
}

// primitiveFitInit(cfgJSON, rgba, alpha) -> {ok,total,steps:[mode,...]} | {error}
func fitInitFunc() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return `{"error":"missing arguments"}`
		}
		var cfg fitConfig
		if err := json.Unmarshal([]byte(args[0].String()), &cfg); err != nil {
			return `{"error":"config parse"}`
		}
		rgbaArray := args[1]
		alphaArray := args[2]
		if cfg.WorkW < 1 || cfg.WorkH < 1 {
			return `{"error":"invalid dimensions"}`
		}
		rgba := make([]byte, rgbaArray.Get("length").Int())
		js.CopyBytesToGo(rgba, rgbaArray)
		if len(rgba) != cfg.WorkW*cfg.WorkH*4 {
			return `{"error":"rgba size mismatch"}`
		}
		var alpha []byte
		if alphaArray.InstanceOf(js.Global().Get("Uint8Array")) && alphaArray.Get("length").Int() == cfg.WorkW*cfg.WorkH {
			alpha = make([]byte, cfg.WorkW*cfg.WorkH)
			js.CopyBytesToGo(alpha, alphaArray)
		} else {
			alpha = make([]byte, cfg.WorkW*cfg.WorkH)
			for i := range alpha {
				alpha[i] = 255
			}
		}

		s := buildSession(cfg, rgba, alpha)
		session = s
		steps := []int{}
		for _, pair := range s.configs {
			for i := 0; i < pair[1]; i++ {
				steps = append(steps, pair[0])
			}
		}
		return marshalOrError(map[string]interface{}{
			"ok":    true,
			"total": s.total,
			"steps": steps,
		}, "init: ")
	})
}

// primitiveFitState() -> {rgba, score, done} (rgba returned as a copy)
func fitStateFunc() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if session == nil {
			return map[string]interface{}{"error": "no session"}
		}
		pix := session.model.Current.Pix
		out := js.Global().Get("Uint8Array").New(len(pix))
		js.CopyBytesToJS(out, pix)
		return map[string]interface{}{
			"rgba":  out,
			"score": session.model.Score,
			"done":  session.done,
		}
	})
}

// primitiveFitSearch(currentRGBA|null, score, mode, wm, nonce) -> candidateJSON
// currentRGBA nil → search from this instance's own model.Current (main path)
func fitSearchFunc() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if session == nil {
			return `{"error":"no session"}`
		}
		if len(args) < 5 {
			return `{"error":"missing arguments"}`
		}
		var current *image.RGBA
		score := args[1].Float()
		if args[0].InstanceOf(js.Global().Get("Uint8Array")) && args[0].Get("length").Int() > 0 {
			w, h := session.cfg.WorkW, session.cfg.WorkH
			pix := make([]byte, args[0].Get("length").Int())
			js.CopyBytesToGo(pix, args[0])
			if len(pix) != w*h*4 {
				return `{"error":"current size mismatch"}`
			}
			current = &image.RGBA{Pix: pix, Stride: w * 4, Rect: image.Rect(0, 0, w, h)}
		} else {
			current = session.model.Current
		}
		mode := args[2].Int()
		wm := args[3].Int()
		nonce := int64(args[4].Int())
		if wm < 1 {
			wm = 1
		}

		worker := session.model.Workers[0]
		// decorrelate this instance's random stream from the other searchers
		worker.Rnd.Seed(time.Now().UnixNano() + nonce*7919)
		worker.Init(current, score)
		state := worker.BestHillClimbState(primitive.ShapeType(mode), 0, 1000, 100, wm)
		if state == nil || state.Shape == nil {
			return `{"error":"no candidate"}`
		}
		return marshalOrError(exportCandidate(state), "search: ")
	})
}

// primitiveFitApply(candidateJSON) -> {rgba, score, done, total, error?}
func fitApplyFunc() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if session == nil {
			return map[string]interface{}{"error": "no session"}
		}
		if len(args) < 1 {
			return map[string]interface{}{"error": "missing candidate"}
		}
		var cj candidateJSON
		if err := json.Unmarshal([]byte(args[0].String()), &cj); err != nil {
			return map[string]interface{}{"error": "candidate parse"}
		}
		shape := buildShape(session.model.Workers[0], cj)
		if shape == nil {
			return map[string]interface{}{"error": "unknown shape type: " + cj.Type}
		}
		session.model.Add(shape, cj.Alpha)
		session.done++

		pix := session.model.Current.Pix
		out := js.Global().Get("Uint8Array").New(len(pix))
		js.CopyBytesToJS(out, pix)
		return map[string]interface{}{
			"rgba":  out,
			"score": session.model.Score,
			"done":  session.done,
			"total": session.total,
		}
	})
}

// primitiveFitFinish() -> fitResult JSON; clears the session
func fitFinishFunc() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if session == nil {
			return `{"error":"no session"}`
		}
		result := finalizeSession(session)
		session = nil
		return marshalOrError(result, "finish: ")
	})
}

func main() {
	js.Global().Set("primitiveFit", fitFunc())
	js.Global().Set("primitiveFitInit", fitInitFunc())
	js.Global().Set("primitiveFitState", fitStateFunc())
	js.Global().Set("primitiveFitSearch", fitSearchFunc())
	js.Global().Set("primitiveFitApply", fitApplyFunc())
	js.Global().Set("primitiveFitFinish", fitFinishFunc())
	// keep the Go runtime alive so the exported function stays callable
	select {}
}
