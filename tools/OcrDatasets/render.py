"""Verified OFL font runs shaped with HarfBuzz and rasterized by FreeType."""
import hashlib
import json
from pathlib import Path

import freetype
import numpy as np
import regex
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from PIL import Image, ImageFilter


class Unsupported(ValueError):
    pass


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Renderer:
    def __init__(self, manifest):
        if str(manifest).startswith(('\\\\', '//')):
            raise ValueError('Network font paths are forbidden')
        self.manifest = Path(manifest).resolve()
        data = json.loads(self.manifest.read_text(encoding='utf-8-sig'))
        self.fonts = data['fonts'] if isinstance(data, dict) else data
        self.loaded = []
        for record in self.fonts:
            for key in ('file', 'license_file'):
                value = record[key]
                if not isinstance(value, str) or Path(value).name != value or '/' in value or '\\' in value or ':' in value:
                    raise ValueError('Font assets must be local direct files')
            if record.get('source_commit') is not None:
                commit = record['source_commit']
                url = record.get('source_url', '')
                if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit) or not url.startswith(('https://raw.githubusercontent.com/notofonts/', 'https://raw.githubusercontent.com/googlefonts/', 'https://github.com/notofonts/', 'https://github.com/googlefonts/')) or commit not in url:
                    raise ValueError('Font provenance must reference an official pinned source')
            path = self.manifest.parent / record['file']
            license_path = self.manifest.parent / record['license_file']
            if file_hash(path) != record['sha256'] or file_hash(license_path) != record['license_sha256']:
                raise ValueError('Font or license hash mismatch')
            if record.get('license') != 'OFL-1.1' or 'SIL OPEN FONT LICENSE' not in license_path.read_text(encoding='utf-8-sig').upper():
                raise ValueError('Only verified OFL fonts are supported')
            raw = path.read_bytes()
            tt = TTFont(path)
            cmap = set(tt.getBestCmap())
            color = any(table in tt for table in ('CBDT', 'COLR', 'sbix'))
            bitmaps = None
            if "CBDT" in tt:
                strike = tt["CBLC"].strikes[0].bitmapSizeTable
                bitmaps = (tt, tt["CBDT"].strikeData[0], strike.ppemY)
            else:
                tt.close()
            face = hb.Face(raw)
            font = hb.Font(face)
            hb.ot_font_set_funcs(font)
            font.scale = (face.upem, face.upem)
            self.loaded.append((record, cmap, color, font, freetype.Face(str(path)), face.upem, bitmaps))
        import PIL
        self.versions = {'pillow': PIL.__version__, 'harfbuzz': hb.version_string(), 'freetype': '.'.join(map(str, freetype.version())), 'regex': regex.__version__}

    def shape(self, text, emoji=False):
        runs = []
        for cluster in regex.findall(r'\X', text):
            required = {ord(c) for c in cluster if c not in ('\ufe0f', '\ufe0e', '\u200d')}
            choices = [f for f in self.loaded if required <= f[1] and (f[2] if emoji else not f[2])]
            if not choices:
                raise Unsupported('missing-cluster')
            selected = choices[0]
            if runs and runs[-1][1] is selected and not emoji:
                runs[-1] = (runs[-1][0] + cluster, selected)
            else:
                runs.append((cluster, selected))
        shaped = []
        for run, font_info in runs:
            buf = hb.Buffer()
            buf.add_str(run)
            buf.guess_segment_properties()
            hb.shape(font_info[3], buf)
            infos, positions = buf.glyph_infos, buf.glyph_positions
            if not infos or any(info.codepoint == 0 for info in infos):
                raise Unsupported('notdef')
            if emoji and len(infos) != 1:
                raise Unsupported('uncomposed-emoji')
            shaped.append((font_info, infos, positions))
        return shaped

    def render(self, text, size=28, dark=False, scale=1.0, blur=0.0, jpeg=None, emoji=False):
        shaped = self.shape(text, emoji)
        width = max(16, int(sum(p.x_advance / f[5] * size for f, _, positions in shaped for p in positions)) + size * 4)
        canvas = Image.new('RGBA', (width, size * 5), (0, 0, 0, 0))
        pen_x, baseline = float(size * 2), size * 3
        used = []
        glyph_count = 0
        color_count = 0
        for info, glyphs, positions in shaped:
            record, _, color, _, face, upem, bitmaps = info
            used.append(record['id'])
            if face.is_scalable:
                face.set_pixel_sizes(0, size)
                ratio = 1.0
            else:
                face.select_size(0)
                ratio = size / (face.size.y_ppem or face.available_sizes[0].height)
            for glyph, pos in zip(glyphs, positions):
                if color and bitmaps:
                    from io import BytesIO
                    tt, strike, ppem = bitmaps
                    name = tt.getGlyphName(glyph.codepoint)
                    if name not in strike:
                        raise Unsupported('missing-color-bitmap')
                    entry = strike[name]
                    glyph_image = Image.open(BytesIO(entry.imageData)).convert('RGBA')
                    arr = np.asarray(glyph_image)
                    if not np.any((arr[:, :, 0] != arr[:, :, 1]) & (arr[:, :, 3] > 0)):
                        raise Unsupported('missing-color')
                    color_count += 1
                    ratio = size / ppem
                    glyph_image = glyph_image.resize((max(1, round(glyph_image.width * ratio)), max(1, round(glyph_image.height * ratio))), Image.Resampling.LANCZOS)
                    x = round(pen_x + pos.x_offset / upem * size + entry.metrics.BearingX * ratio)
                    y = round(baseline - pos.y_offset / upem * size - entry.metrics.BearingY * ratio)
                    if x < 0 or y < 0 or x + glyph_image.width > canvas.width or y + glyph_image.height > canvas.height:
                        raise Unsupported('clipping')
                    canvas.alpha_composite(glyph_image, (x, y))
                    glyph_count += 1
                    pen_x += pos.x_advance / upem * size
                    continue
                face.load_glyph(glyph.codepoint, freetype.FT_LOAD_RENDER | (freetype.FT_LOAD_COLOR if color else 0))
                bitmap = face.glyph.bitmap
                if bitmap.width and bitmap.rows:
                    raw = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, abs(bitmap.pitch))
                    if bitmap.pitch < 0:
                        raw = raw[::-1]
                    if bitmap.pixel_mode == freetype.FT_PIXEL_MODE_BGRA:
                        arr = raw[:, :bitmap.width * 4].reshape(bitmap.rows, bitmap.width, 4)[:, :, [2, 1, 0, 3]].copy()
                        # FreeType color pixels are premultiplied; Pillow composites straight alpha.
                        alpha = arr[:, :, 3].astype(float)
                        for channel in range(3):
                            arr[:, :, channel] = np.minimum(255, np.divide(arr[:, :, channel].astype(float) * 255, alpha, out=np.zeros_like(alpha), where=alpha > 0)).astype('uint8')
                        color_count += int(np.any((arr[:, :, 0] != arr[:, :, 1]) & (alpha > 0)))
                        glyph_image = Image.fromarray(arr, 'RGBA')
                    elif bitmap.pixel_mode == freetype.FT_PIXEL_MODE_GRAY:
                        ink = 235 if dark else 25
                        glyph_image = Image.new('RGBA', (bitmap.width, bitmap.rows), (ink, ink, ink, 0))
                        glyph_image.putalpha(Image.fromarray(raw[:, :bitmap.width].copy()))
                    else:
                        raise Unsupported('unsupported-bitmap-mode')
                    if ratio != 1:
                        glyph_image = glyph_image.resize((max(1, round(glyph_image.width * ratio)), max(1, round(glyph_image.height * ratio))), Image.Resampling.LANCZOS)
                    x = round(pen_x + pos.x_offset / upem * size + face.glyph.bitmap_left * ratio)
                    y = round(baseline - pos.y_offset / upem * size - face.glyph.bitmap_top * ratio)
                    if x < 0 or y < 0 or x + glyph_image.width > canvas.width or y + glyph_image.height > canvas.height:
                        raise Unsupported('clipping')
                    canvas.alpha_composite(glyph_image, (x, y))
                    glyph_count += 1
                pen_x += pos.x_advance / upem * size
        bounds = canvas.getbbox()
        if not bounds:
            raise Unsupported('empty-render')
        if emoji and not color_count:
            raise Unsupported('missing-color')
        # Retain surrounding advance space; do not crop leading/trailing spaces away.
        image = Image.new('RGB', canvas.size, (35, 39, 46) if dark else (246, 246, 242))
        image.paste(canvas, mask=canvas.getchannel('A'))
        alpha = canvas.getchannel('A')
        crop = (0, max(0, bounds[1] - 5), min(canvas.width, max(round(pen_x) + size * 2, bounds[2] + 5)), min(canvas.height, bounds[3] + 5))
        image, alpha = image.crop(crop), alpha.crop(crop)
        dims = (max(8, round(image.width * scale)), max(8, round(image.height * scale)))
        image = image.resize(dims, Image.Resampling.LANCZOS)
        alpha = alpha.resize(dims, Image.Resampling.LANCZOS)
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(blur))
            alpha = alpha.filter(ImageFilter.GaussianBlur(blur))
        if jpeg:
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=jpeg, subsampling=0)
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB')
        bbox = alpha.getbbox()
        return image, [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], sorted(set(used)), {'glyphs': glyph_count, 'colored_glyphs': color_count, 'composed': emoji, 'bounds_basis': 'transformed-raster-alpha'}
