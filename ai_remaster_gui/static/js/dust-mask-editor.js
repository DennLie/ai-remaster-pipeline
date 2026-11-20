/* Interactive lossless bitmask canvas editor for Dust & Scratches.
 *
 * Packed bit flags match ai_remaster_gui/dust_mask_edits.py identically:
 *   BIT_SCRATCH_ADD      = 1   (0x01)
 *   BIT_SCRATCH_REJECT   = 2   (0x02)
 *   BIT_DUST_ADD         = 4   (0x04)
 *   BIT_DUST_REJECT      = 8   (0x08)
 *   BIT_PROTECT_INTACT   = 16  (0x10)
 *   BIT_UNCERTAIN_REJECT = 32  (0x20)
 *   BIT_REVIEWED         = 64  (0x40)
 *
 * Bitmasks are stored and transported exclusively as contiguous row-major
 * Uint8Array buffers <-> raw base64 bytes, never via premultiplied RGBA pixels.
 */

const BIT_SCRATCH_ADD = 1;
const BIT_SCRATCH_REJECT = 2;
const BIT_DUST_ADD = 4;
const BIT_DUST_REJECT = 8;
const BIT_PROTECT_INTACT = 16;
const BIT_UNCERTAIN_REJECT = 32;
const BIT_REVIEWED = 64;
const KNOWN_EDIT_BITS_MASK =
  BIT_SCRATCH_ADD |
  BIT_SCRATCH_REJECT |
  BIT_DUST_ADD |
  BIT_DUST_REJECT |
  BIT_PROTECT_INTACT |
  BIT_UNCERTAIN_REJECT |
  BIT_REVIEWED;

const DUST_EDIT_TOOLS = [
  'scratch_add',
  'scratch_reject',
  'dust_add',
  'dust_reject',
  'uncertain_reject',
  'reject_all',
  'protect',
  'unprotect',
  'clear_edits',
  'mark_reviewed',
];

const DUST_EDITOR_MAX_UNDO = 30;
const DUST_EDITOR_AUTOSAVE_DEBOUNCE_MS = 180;
const _B64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function _encodeBase64Fallback(bytes) {
  let out = '';
  const len = bytes.length;
  for (let i = 0; i < len; i += 3) {
    const b0 = bytes[i];
    const has1 = i + 1 < len;
    const has2 = i + 2 < len;
    const b1 = has1 ? bytes[i + 1] : 0;
    const b2 = has2 ? bytes[i + 2] : 0;
    const trip = (b0 << 16) | (b1 << 8) | b2;
    out +=
      _B64_ALPHABET[(trip >> 18) & 0x3f] +
      _B64_ALPHABET[(trip >> 12) & 0x3f] +
      (has1 ? _B64_ALPHABET[(trip >> 6) & 0x3f] : '=') +
      (has2 ? _B64_ALPHABET[trip & 0x3f] : '=');
  }
  return out;
}

function _decodeBase64Fallback(b64) {
  if (b64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(b64)) {
    throw new Error('Invalid base64 bitmask payload.');
  }
  if (b64.length === 0) return new Uint8Array(0);
  const pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
  const outLen = (b64.length / 4) * 3 - pad;
  const out = new Uint8Array(outLen);
  let p = 0;
  for (let i = 0; i < b64.length; i += 4) {
    const c0 = _B64_ALPHABET.indexOf(b64[i]);
    const c1 = _B64_ALPHABET.indexOf(b64[i + 1]);
    const c2 = b64[i + 2] === '=' ? 0 : _B64_ALPHABET.indexOf(b64[i + 2]);
    const c3 = b64[i + 3] === '=' ? 0 : _B64_ALPHABET.indexOf(b64[i + 3]);
    if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) {
      throw new Error('Invalid base64 bitmask payload.');
    }
    const trip = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
    if (p < outLen) out[p++] = (trip >> 16) & 0xff;
    if (p < outLen) out[p++] = (trip >> 8) & 0xff;
    if (p < outLen) out[p++] = trip & 0xff;
  }
  return out;
}

function _isUint8Array(v) {
  return Boolean(
    v &&
      ArrayBuffer.isView(v) &&
      v.BYTES_PER_ELEMENT === 1 &&
      Object.prototype.toString.call(v) === '[object Uint8Array]'
  );
}

function _isUint8MaskView(v) {
  if (!v || !ArrayBuffer.isView(v) || v.BYTES_PER_ELEMENT !== 1) return false;
  const tag = Object.prototype.toString.call(v);
  return tag === '[object Uint8Array]' || tag === '[object Uint8ClampedArray]';
}

function dustMaskEditorIsUint8(v) {
  return _isUint8Array(v);
}

function dustBitmaskToBase64(uint8Array) {
  let bytes;
  if (_isUint8Array(uint8Array)) {
    bytes = uint8Array;
  } else if (ArrayBuffer.isView(uint8Array)) {
    bytes = new Uint8Array(uint8Array.buffer, uint8Array.byteOffset, uint8Array.byteLength);
  } else {
    throw new Error('dustBitmaskToBase64 expects a Uint8Array.');
  }
  if (bytes.length === 0) return '';
  if (typeof Buffer !== 'undefined' && typeof Buffer.from === 'function') {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
  }
  if (typeof btoa === 'function') {
    const chunkSize = 0x8000;
    const chunks = [];
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const slice = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
      chunks.push(String.fromCharCode.apply(null, slice));
    }
    return btoa(chunks.join(''));
  }
  return _encodeBase64Fallback(bytes);
}

function dustBase64ToBitmask(b64, width, height) {
  const w = Number(width);
  const h = Number(height);
  if (!Number.isInteger(w) || w <= 0 || !Number.isInteger(h) || h <= 0) {
    throw new Error(`Invalid frame dimensions for bitmask decode: ${width}x${height}`);
  }
  const expectedLen = w * h;
  if (b64 === '' || b64 === null || b64 === undefined) {
    return new Uint8Array(expectedLen);
  }
  if (typeof b64 !== 'string') {
    throw new Error('edit_mask_b64 must be a base64 string.');
  }
  let bytes;
  if (typeof Buffer !== 'undefined' && typeof Buffer.from === 'function') {
    if (b64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(b64)) {
      throw new Error('Invalid base64 bitmask payload.');
    }
    const buf = Buffer.from(b64, 'base64');
    bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  } else if (typeof atob === 'function') {
    let bin;
    try {
      bin = atob(b64);
    } catch (_err) {
      throw new Error('Invalid base64 bitmask payload.');
    }
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) {
      bytes[i] = bin.charCodeAt(i);
    }
  } else {
    bytes = _decodeBase64Fallback(b64);
  }
  if (bytes.length !== expectedLen) {
    throw new Error(
      `Decoded bitmask byte length ${bytes.length} does not match ${w}x${h} (${expectedLen} bytes).`
    );
  }
  return new Uint8Array(bytes);
}

function dustBitmaskHasEdits(uint8Array) {
  if (!_isUint8Array(uint8Array)) return false;
  for (let i = 0; i < uint8Array.length; i += 1) {
    if (uint8Array[i] !== 0) return true;
  }
  return false;
}

function dustComputeFrameStats(bitmaskU8) {
  const stats = {
    scratch_add_pixels: 0,
    scratch_reject_pixels: 0,
    dust_add_pixels: 0,
    dust_reject_pixels: 0,
    uncertain_reject_pixels: 0,
    protect_intact_pixels: 0,
    reviewed_pixels: 0,
    edited_pixels: 0,
  };
  if (!_isUint8Array(bitmaskU8)) return stats;
  for (let i = 0; i < bitmaskU8.length; i += 1) {
    const v = bitmaskU8[i];
    if (v === 0) continue;
    stats.edited_pixels += 1;
    if (v & BIT_SCRATCH_ADD) stats.scratch_add_pixels += 1;
    if (v & BIT_SCRATCH_REJECT) stats.scratch_reject_pixels += 1;
    if (v & BIT_DUST_ADD) stats.dust_add_pixels += 1;
    if (v & BIT_DUST_REJECT) stats.dust_reject_pixels += 1;
    if (v & BIT_UNCERTAIN_REJECT) stats.uncertain_reject_pixels += 1;
    if (v & BIT_PROTECT_INTACT) stats.protect_intact_pixels += 1;
    if (v & BIT_REVIEWED) stats.reviewed_pixels += 1;
  }
  return stats;
}

function _applyBitTransitionAt(arr, idx, tool) {
  const v = arr[idx];
  switch (tool) {
    case 'scratch_add':
      arr[idx] = (v & ~BIT_SCRATCH_REJECT) | BIT_SCRATCH_ADD;
      break;
    case 'scratch_reject':
      arr[idx] = (v & ~BIT_SCRATCH_ADD) | BIT_SCRATCH_REJECT;
      break;
    case 'dust_add':
      arr[idx] = (v & ~BIT_DUST_REJECT) | BIT_DUST_ADD;
      break;
    case 'dust_reject':
      arr[idx] = (v & ~BIT_DUST_ADD) | BIT_DUST_REJECT;
      break;
    case 'uncertain_reject':
      arr[idx] = v | BIT_UNCERTAIN_REJECT;
      break;
    case 'reject_all':
      arr[idx] =
        (v & ~(BIT_SCRATCH_ADD | BIT_DUST_ADD)) |
        (BIT_SCRATCH_REJECT | BIT_DUST_REJECT | BIT_UNCERTAIN_REJECT);
      break;
    case 'protect':
      arr[idx] = v | BIT_PROTECT_INTACT;
      break;
    case 'unprotect':
      arr[idx] = v & ~BIT_PROTECT_INTACT;
      break;
    case 'clear_edits':
      arr[idx] = 0;
      break;
    case 'mark_reviewed':
      arr[idx] = v | BIT_REVIEWED;
      break;
    default:
      throw new Error(`Unsupported stroke tool: ${tool}`);
  }
}

function dustApplyStrokeTransition(bitmaskU8, pixelIndicesOrMask, tool, options = {}) {
  if (!_isUint8Array(bitmaskU8)) {
    throw new Error('dustApplyStrokeTransition expects a Uint8Array bitmask.');
  }
  const normTool = String(tool || '').trim().toLowerCase();
  if (!DUST_EDIT_TOOLS.includes(normTool)) {
    throw new Error(`Unsupported stroke tool '${tool}'.`);
  }
  const out = options && options.inPlace ? bitmaskU8 : new Uint8Array(bitmaskU8);
  if (pixelIndicesOrMask == null) return out;

  if (
    _isUint8MaskView(pixelIndicesOrMask) &&
    pixelIndicesOrMask.length === out.length
  ) {
    for (let i = 0; i < out.length; i += 1) {
      if (pixelIndicesOrMask[i] !== 0) _applyBitTransitionAt(out, i, normTool);
    }
    return out;
  }

  if (
    Array.isArray(pixelIndicesOrMask) &&
    pixelIndicesOrMask.length === out.length &&
    pixelIndicesOrMask.some(val => typeof val === 'boolean')
  ) {
    for (let i = 0; i < out.length; i += 1) {
      if (pixelIndicesOrMask[i]) _applyBitTransitionAt(out, i, normTool);
    }
    return out;
  }

  for (const rawIdx of pixelIndicesOrMask) {
    const idx = Number(rawIdx);
    if (Number.isInteger(idx) && idx >= 0 && idx < out.length) {
      _applyBitTransitionAt(out, idx, normTool);
    }
  }
  return out;
}

function dustCanvasClientToPixel(canvas, clientX, clientY) {
  if (!canvas || typeof canvas.getBoundingClientRect !== 'function') return null;
  const rect = canvas.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0 && canvas.width > 0 && canvas.height > 0)) {
    return null;
  }
  const x = Math.max(0, Math.min(canvas.width - 1, Math.floor(((clientX - rect.left) * canvas.width) / rect.width)));
  const y = Math.max(0, Math.min(canvas.height - 1, Math.floor(((clientY - rect.top) * canvas.height) / rect.height)));
  return { x, y };
}

function dustClientPointToPixel(canvas, clientX, clientY) {
  return dustCanvasClientToPixel(canvas, clientX, clientY);
}

function _distSqPointToSegment(px, py, x0, y0, x1, y1) {
  const vx = x1 - x0;
  const vy = y1 - y0;
  const lenSq = vx * vx + vy * vy;
  if (lenSq <= 0) {
    const dx = px - x0;
    const dy = py - y0;
    return dx * dx + dy * dy;
  }
  let t = ((px - x0) * vx + (py - y0) * vy) / lenSq;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  const projX = x0 + t * vx;
  const projY = y0 + t * vy;
  const dx = px - projX;
  const dy = py - projY;
  return dx * dx + dy * dy;
}

function dustCollectBrushSegmentPixels(x0, y0, x1, y1, width, height, radius) {
  const w = Math.max(0, Math.floor(Number(width) || 0));
  const h = Math.max(0, Math.floor(Number(height) || 0));
  if (w === 0 || h === 0) return [];
  const cx0 = Math.max(0, Math.min(w - 1, Math.round(Number(x0) || 0)));
  const cy0 = Math.max(0, Math.min(h - 1, Math.round(Number(y0) || 0)));
  const cx1 = Math.max(0, Math.min(w - 1, Math.round(Number(x1) || 0)));
  const cy1 = Math.max(0, Math.min(h - 1, Math.round(Number(y1) || 0)));
  const r = Math.max(0, Number(radius) || 0);
  const indices = [];

  if (r <= 0) {
    // Exact integer Bresenham line for single-pixel editing.
    let x = cx0;
    let y = cy0;
    const dx = Math.abs(cx1 - cx0);
    const dy = Math.abs(cy1 - cy0);
    const sx = cx0 < cx1 ? 1 : -1;
    const sy = cy0 < cy1 ? 1 : -1;
    let err = dx - dy;
    while (true) {
      indices.push(y * w + x);
      if (x === cx1 && y === cy1) break;
      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x += sx;
      }
      if (e2 < dx) {
        err += dx;
        y += sy;
      }
    }
    return indices;
  }

  const rSq = r * r;
  const minX = Math.max(0, Math.floor(Math.min(cx0, cx1) - r));
  const maxX = Math.min(w - 1, Math.ceil(Math.max(cx0, cx1) + r));
  const minY = Math.max(0, Math.floor(Math.min(cy0, cy1) - r));
  const maxY = Math.min(h - 1, Math.ceil(Math.max(cy0, cy1) + r));

  for (let y = minY; y <= maxY; y += 1) {
    const rowOff = y * w;
    for (let x = minX; x <= maxX; x += 1) {
      if (_distSqPointToSegment(x, y, cx0, cy0, cx1, cy1) <= rSq) {
        indices.push(rowOff + x);
      }
    }
  }
  return indices;
}

function paintBrushSegment(bitmaskU8, width, height, x0, y0, x1, y1, radius, tool, options = {}) {
  if (!_isUint8Array(bitmaskU8)) {
    throw new Error('paintBrushSegment expects a Uint8Array bitmask.');
  }
  const w = Math.max(0, Math.floor(Number(width) || 0));
  const h = Math.max(0, Math.floor(Number(height) || 0));
  if (bitmaskU8.length !== w * h) {
    throw new Error(`Bitmask length ${bitmaskU8.length} does not match ${w}x${h}.`);
  }
  const normTool = String(tool || '').trim().toLowerCase();
  if (!DUST_EDIT_TOOLS.includes(normTool)) {
    throw new Error(`Unsupported stroke tool '${tool}'.`);
  }
  const inPlace = options && options.inPlace !== undefined ? Boolean(options.inPlace) : true;
  const out = inPlace ? bitmaskU8 : new Uint8Array(bitmaskU8);
  const cx0 = Math.max(0, Math.min(w - 1, Math.round(Number(x0) || 0)));
  const cy0 = Math.max(0, Math.min(h - 1, Math.round(Number(y0) || 0)));
  const cx1 = Math.max(0, Math.min(w - 1, Math.round(Number(x1) || 0)));
  const cy1 = Math.max(0, Math.min(h - 1, Math.round(Number(y1) || 0)));
  const r = Math.max(0, Number(radius) || 0);

  if (r <= 0) {
    const pixels = dustCollectBrushSegmentPixels(cx0, cy0, cx1, cy1, w, h, 0);
    for (let i = 0; i < pixels.length; i += 1) {
      _applyBitTransitionAt(out, pixels[i], normTool);
    }
    if (options && typeof options === 'object') {
      options.lastDirtyRect = {
        minX: Math.min(cx0, cx1),
        minY: Math.min(cy0, cy1),
        maxX: Math.max(cx0, cx1),
        maxY: Math.max(cy0, cy1),
      };
    }
    return out;
  }

  const rSq = r * r;
  const minX = Math.max(0, Math.floor(Math.min(cx0, cx1) - r));
  const maxX = Math.min(w - 1, Math.ceil(Math.max(cx0, cx1) + r));
  const minY = Math.max(0, Math.floor(Math.min(cy0, cy1) - r));
  const maxY = Math.min(h - 1, Math.ceil(Math.max(cy0, cy1) + r));

  for (let y = minY; y <= maxY; y += 1) {
    const rowOff = y * w;
    for (let x = minX; x <= maxX; x += 1) {
      if (_distSqPointToSegment(x, y, cx0, cy0, cx1, cy1) <= rSq) {
        _applyBitTransitionAt(out, rowOff + x, normTool);
      }
    }
  }
  if (options && typeof options === 'object') {
    options.lastDirtyRect = { minX, minY, maxX, maxY };
  }
  return out;
}

function dustExtractBinaryPlane(img, width, height) {
  if (!img || !img.naturalWidth || !img.naturalHeight || typeof document === 'undefined') {
    return null;
  }
  const off = document.createElement('canvas');
  off.width = width;
  off.height = height;
  const ctx = off.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0, width, height);
  const data = ctx.getImageData(0, 0, width, height).data;
  const out = new Uint8Array(width * height);
  for (let p = 0, i = 0; p < out.length; p += 1, i += 4) {
    out[p] = data[i] >= 128 ? 255 : 0;
  }
  return out;
}

function _isProtectEdge(bitmask, x, y, w, h) {
  const p = y * w + x;
  if (x === 0 || x === w - 1 || y === 0 || y === h - 1) return true;
  if ((bitmask[p - 1] & BIT_PROTECT_INTACT) === 0) return true;
  if ((bitmask[p + 1] & BIT_PROTECT_INTACT) === 0) return true;
  if ((bitmask[p - w] & BIT_PROTECT_INTACT) === 0) return true;
  if ((bitmask[p + w] & BIT_PROTECT_INTACT) === 0) return true;
  return false;
}

function _fillOverlayImageData(rgbaData, params) {
  const {
    width: w,
    height: h,
    mode = 'overlay',
    bitmask = null,
    autoScratch = null,
    autoDust = null,
    autoUncertain = null,
    autoVetoed = null,
    autoCombined = null,
    opacity = 0.55,
    layers = { auto: true, human: true, protect: true, vetoed: true, rejected: true },
    overlayOnly = false,
    dirtyRect = null,
  } = params;

  const showAuto = layers.auto !== false;
  const showHuman = layers.human !== false;
  const showProtect = layers.protect !== false;
  const showVetoed = layers.vetoed !== false;
  const showRejected = layers.rejected !== false;

  const clampedOp = Math.max(0.05, Math.min(1, Number(opacity) || 0.55));
  const aAlpha = Math.round(255 * clampedOp);
  const hAlpha = Math.min(255, Math.round(255 * Math.max(0.35, clampedOp * 1.25)));
  const pAlpha = Math.min(255, Math.round(255 * Math.max(0.4, clampedOp * 1.15)));
  const vAlpha = Math.round(255 * Math.max(0.25, clampedOp * 0.85));

  const minX = dirtyRect ? Math.max(0, dirtyRect.minX) : 0;
  const maxX = dirtyRect ? Math.min(w - 1, dirtyRect.maxX) : w - 1;
  const minY = dirtyRect ? Math.max(0, dirtyRect.minY) : 0;
  const maxY = dirtyRect ? Math.min(h - 1, dirtyRect.maxY) : h - 1;

  const opaqueBg = !overlayOnly && ['mask', 'scratch', 'dust', 'uncertain', 'vetoed'].includes(mode);

  for (let y = minY; y <= maxY; y += 1) {
    const rowOff = y * w;
    for (let x = minX; x <= maxX; x += 1) {
      const p = rowOff + x;
      const i = p * 4;
      const b = bitmask ? bitmask[p] : 0;
      const prot = (b & BIT_PROTECT_INTACT) !== 0;
      const sAdd = (b & BIT_SCRATCH_ADD) !== 0;
      const sRej = (b & BIT_SCRATCH_REJECT) !== 0;
      const dAdd = (b & BIT_DUST_ADD) !== 0;
      const dRej = (b & BIT_DUST_REJECT) !== 0;
      const uRej = (b & BIT_UNCERTAIN_REJECT) !== 0;

      const aS = autoScratch ? autoScratch[p] >= 128 : false;
      const aD = autoDust ? autoDust[p] >= 128 : false;
      const aU = autoUncertain ? autoUncertain[p] >= 128 : false;
      const aV = autoVetoed ? autoVetoed[p] >= 128 : false;
      const aC = autoCombined ? autoCombined[p] >= 128 : false;

      if (mode === 'overlay' || overlayOnly) {
        if (prot && showProtect) {
          const edge = _isProtectEdge(bitmask, x, y, w, h);
          const stripe = ((x + y) % 6) < 2;
          rgbaData[i] = edge ? 48 : stripe ? 42 : 24;
          rgbaData[i + 1] = edge ? 248 : stripe ? 224 : 188;
          rgbaData[i + 2] = edge ? 152 : stripe ? 136 : 108;
          rgbaData[i + 3] = edge ? Math.min(255, pAlpha + 72) : stripe ? Math.min(255, pAlpha + 26) : pAlpha;
          continue;
        }
        if (!prot && !sRej && sAdd && showHuman) {
          rgbaData[i] = 255;
          rgbaData[i + 1] = 64;
          rgbaData[i + 2] = 88;
          rgbaData[i + 3] = hAlpha;
          continue;
        }
        if (!prot && !dRej && dAdd && showHuman) {
          rgbaData[i] = 255;
          rgbaData[i + 1] = 142;
          rgbaData[i + 2] = 24;
          rgbaData[i + 3] = hAlpha;
          continue;
        }
        const effS = !prot && !sRej && aS;
        const effD = !prot && !dRej && aD;
        const effU = !prot && !uRej && !(sAdd && !sRej) && !(dAdd && !dRej) && aU;
        if (effS && showAuto) {
          rgbaData[i] = 0;
          rgbaData[i + 1] = 206;
          rgbaData[i + 2] = 255;
          rgbaData[i + 3] = aAlpha;
          continue;
        }
        if (effD && showAuto) {
          rgbaData[i] = 255;
          rgbaData[i + 1] = 192;
          rgbaData[i + 2] = 28;
          rgbaData[i + 3] = aAlpha;
          continue;
        }
        if (effU && showAuto) {
          rgbaData[i] = 196;
          rgbaData[i + 1] = 112;
          rgbaData[i + 2] = 255;
          rgbaData[i + 3] = aAlpha;
          continue;
        }
        if (!prot && !sRej && !dRej && !uRej && aC && !aS && !aD && !aU && showAuto) {
          rgbaData[i] = 255;
          rgbaData[i + 1] = 0;
          rgbaData[i + 2] = 255;
          rgbaData[i + 3] = aAlpha;
          continue;
        }
        if ((sRej || dRej || uRej) && showRejected) {
          const hatch = ((x - y + 65536) % 5) === 0;
          rgbaData[i] = hatch ? 220 : 48;
          rgbaData[i + 1] = hatch ? 225 : 54;
          rgbaData[i + 2] = hatch ? 232 : 66;
          rgbaData[i + 3] = hatch ? Math.min(255, Math.round(hAlpha * 0.9)) : Math.round(aAlpha * 0.72);
          continue;
        }
        if (aV && showVetoed) {
          rgbaData[i] = 56;
          rgbaData[i + 1] = 178;
          rgbaData[i + 2] = 172;
          rgbaData[i + 3] = vAlpha;
          continue;
        }
        rgbaData[i] = 0;
        rgbaData[i + 1] = 0;
        rgbaData[i + 2] = 0;
        rgbaData[i + 3] = 0;
        continue;
      }

      // Channel modes (mask / scratch / dust / uncertain / vetoed)
      if (prot && showProtect) {
        const stripe = ((x + y) % 6) < 2;
        rgbaData[i] = stripe ? 32 : 14;
        rgbaData[i + 1] = stripe ? 185 : 115;
        rgbaData[i + 2] = stripe ? 105 : 68;
        rgbaData[i + 3] = 255;
        continue;
      }

      if (mode === 'vetoed') {
        if (aV) {
          rgbaData[i] = 88;
          rgbaData[i + 1] = 230;
          rgbaData[i + 2] = 220;
          rgbaData[i + 3] = 255;
        } else {
          rgbaData[i] = 0;
          rgbaData[i + 1] = 0;
          rgbaData[i + 2] = 0;
          rgbaData[i + 3] = opaqueBg ? 255 : 0;
        }
        continue;
      }

      let active = false;
      let rejectedUnder = false;
      if (mode === 'scratch') {
        active = !prot && !sRej && (sAdd || aS);
        rejectedUnder = sRej && aS;
      } else if (mode === 'dust') {
        active = !prot && !dRej && (dAdd || aD);
        rejectedUnder = dRej && aD;
      } else if (mode === 'uncertain') {
        active = !prot && !uRej && !(sAdd && !sRej) && !(dAdd && !dRej) && aU;
        rejectedUnder = uRej && aU;
      } else {
        // Combined effective mask
        const effS = !prot && !sRej && (sAdd || aS);
        const effD = !prot && !dRej && (dAdd || aD);
        const effU = !prot && !uRej && !(sAdd && !sRej) && !(dAdd && !dRej) && aU;
        const effC = !prot && !sRej && !dRej && !uRej && aC && !aS && !aD && !aU;
        active = effS || effD || effU || effC;
        rejectedUnder = !active && ((sRej && aS) || (dRej && aD) || (uRej && aU));
      }

      if (active) {
        if (showHuman && !sRej && sAdd && (mode === 'mask' || mode === 'scratch')) {
          rgbaData[i] = 255;
          rgbaData[i + 1] = 175;
          rgbaData[i + 2] = 185;
          rgbaData[i + 3] = 255;
        } else if (showHuman && !dRej && dAdd && (mode === 'mask' || mode === 'dust')) {
          rgbaData[i] = 255;
          rgbaData[i + 1] = 205;
          rgbaData[i + 2] = 140;
          rgbaData[i + 3] = 255;
        } else {
          rgbaData[i] = 255;
          rgbaData[i + 1] = 255;
          rgbaData[i + 2] = 255;
          rgbaData[i + 3] = 255;
        }
      } else if (rejectedUnder && showRejected) {
        const hatch = ((x - y + 65536) % 5) === 0;
        rgbaData[i] = hatch ? 140 : 40;
        rgbaData[i + 1] = hatch ? 70 : 25;
        rgbaData[i + 2] = hatch ? 80 : 30;
        rgbaData[i + 3] = 255;
      } else {
        rgbaData[i] = 0;
        rgbaData[i + 1] = 0;
        rgbaData[i + 2] = 0;
        rgbaData[i + 3] = opaqueBg ? 255 : 0;
      }
    }
  }
}

function renderDustMaskOverlayCanvas(targetCanvas, params = {}) {
  if (!targetCanvas) return null;
  const w = Number(params.width || targetCanvas.width || 0);
  const h = Number(params.height || targetCanvas.height || 0);
  if (!(w > 0 && h > 0)) return null;

  if (targetCanvas.width !== w) targetCanvas.width = w;
  if (targetCanvas.height !== h) targetCanvas.height = h;
  const ctx = targetCanvas.getContext('2d');
  if (!ctx) return null;

  const mode = params.mode || 'overlay';
  const overlayOnly = Boolean(params.overlayOnly);
  const baseImage = params.baseImage || null;
  const splitActive = Boolean(params.splitActive);
  const splitPos = Math.max(0, Math.min(100, Number(params.splitPos ?? 50)));

  let overlayCanvas = params.overlayCanvas || null;
  let overlayImageData = params.overlayImageData || null;

  if (overlayOnly) {
    if (!overlayImageData || overlayImageData.width !== w || overlayImageData.height !== h) {
      overlayImageData = ctx.createImageData(w, h);
    }
    _fillOverlayImageData(overlayImageData.data, { ...params, width: w, height: h, overlayOnly: true });
    if (params.dirtyRect) {
      const dr = params.dirtyRect;
      const dx = Math.max(0, dr.minX);
      const dy = Math.max(0, dr.minY);
      const dw = Math.min(w, dr.maxX + 1) - dx;
      const dh = Math.min(h, dr.maxY + 1) - dy;
      if (dw > 0 && dh > 0) ctx.putImageData(overlayImageData, 0, 0, dx, dy, dw, dh);
    } else {
      ctx.putImageData(overlayImageData, 0, 0);
    }
    return overlayImageData;
  }

  if (!overlayCanvas && typeof document !== 'undefined') {
    overlayCanvas = document.createElement('canvas');
  }
  if (overlayCanvas) {
    if (overlayCanvas.width !== w) overlayCanvas.width = w;
    if (overlayCanvas.height !== h) overlayCanvas.height = h;
    const octx = overlayCanvas.getContext('2d');
    if (!overlayImageData || overlayImageData.width !== w || overlayImageData.height !== h) {
      overlayImageData = octx.createImageData(w, h);
    }
    _fillOverlayImageData(overlayImageData.data, {
      ...params,
      width: w,
      height: h,
      overlayOnly: mode === 'overlay',
    });
    if (params.dirtyRect) {
      const dr = params.dirtyRect;
      const dx = Math.max(0, dr.minX);
      const dy = Math.max(0, dr.minY);
      const dw = Math.min(w, dr.maxX + 1) - dx;
      const dh = Math.min(h, dr.maxY + 1) - dy;
      if (dw > 0 && dh > 0) octx.putImageData(overlayImageData, 0, 0, dx, dy, dw, dh);
    } else {
      octx.putImageData(overlayImageData, 0, 0);
    }
  }

  ctx.clearRect(0, 0, w, h);

  const drawComposite = () => {
    if (mode === 'input') {
      if (baseImage) ctx.drawImage(baseImage, 0, 0, w, h);
    } else if (mode === 'overlay') {
      if (baseImage) ctx.drawImage(baseImage, 0, 0, w, h);
      if (overlayCanvas) ctx.drawImage(overlayCanvas, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, w, h);
      if (overlayCanvas) ctx.drawImage(overlayCanvas, 0, 0, w, h);
    }
  };

  if (splitActive && baseImage) {
    const splitX = Math.max(0, Math.min(w, Math.round(w * (splitPos / 100))));
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitX, h);
    ctx.clip();
    ctx.drawImage(baseImage, 0, 0, w, h);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, w - splitX, h);
    ctx.clip();
    drawComposite();
    ctx.restore();

    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(splitX - 2, 0, 4, h);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(splitX - 1, 0, 2, h);
  } else {
    drawComposite();
  }

  return { overlayCanvas, overlayImageData };
}

const dustMaskEditor = {
  editMode: false,
  tool: 'scratch_add',
  brushRadius: 3,
  layers: {
    auto: true,
    human: true,
    protect: true,
    vetoed: true,
    rejected: true,
  },
  frame: -1,
  frameCount: 0,
  width: 0,
  height: 0,
  analysisSource: 'stage',
  timelineId: '',
  editRevision: 0,
  zipSha256: '',
  editSignature: '',
  compatible: true,
  compatibilityReason: '',
  editedFrames: [],
  reviewed: false,
  frameStats: null,
  summary: null,
  bitmask: null,
  undoStack: [],
  redoStack: [],
  dirty: false,
  drawing: false,
  activePointerId: null,
  lastPoint: null,
  saving: false,
  loading: false,
  loadingFrame: -1,
  savePromise: Promise.resolve(),
  saveTimer: null,
  loadEpoch: 0,
  statusText: '',
  statusTone: '',
  exportPending: false,
  spacePressed: false,
  keybindingsInstalled: false,
  frameRender: null,
  overlayCanvas: null,
  overlayImageData: null,
};

function dustMaskEditorIsBusy() {
  return Boolean(
    dustMaskEditor.drawing ||
      dustMaskEditor.dirty ||
      dustMaskEditor.saving ||
      dustMaskEditor.exportPending
  );
}

function dustMaskEditorIsEditActive() {
  return Boolean(dustMaskEditor.editMode);
}

function _dustProjectPath() {
  return typeof state !== 'undefined' && state?.project_path ? String(state.project_path) : '';
}

function _dustActiveSourceMode() {
  return typeof dustSourceMode === 'string' && dustSourceMode === 'original' ? 'original' : 'stage';
}

function _dustSetStatus(text, tone = '') {
  dustMaskEditor.statusText = String(text || '');
  dustMaskEditor.statusTone = tone;
  dustMaskEditorSyncUi();
}

function dustMaskEditorResetLocalCache() {
  if (dustMaskEditor.saveTimer) {
    clearTimeout(dustMaskEditor.saveTimer);
    dustMaskEditor.saveTimer = null;
  }
  dustMaskEditor.loadEpoch += 1;
  dustMaskEditor.loading = false;
  dustMaskEditor.loadingFrame = -1;
  dustMaskEditor.analysisSource = _dustActiveSourceMode();
  dustMaskEditor.frame = -1;
  dustMaskEditor.width = 0;
  dustMaskEditor.height = 0;
  dustMaskEditor.bitmask = null;
  dustMaskEditor.undoStack = [];
  dustMaskEditor.redoStack = [];
  dustMaskEditor.dirty = false;
  dustMaskEditor.drawing = false;
  dustMaskEditor.activePointerId = null;
  dustMaskEditor.lastPoint = null;
  dustMaskEditor.frameStats = null;
  dustMaskEditor.frameRender = null;
  dustMaskEditor.overlayImageData = null;
}

function dustMaskEditorResetProject() {
  dustMaskEditorResetLocalCache();
  dustMaskEditor.editMode = false;
  dustMaskEditor.spacePressed = false;
  dustMaskEditor.analysisSource = _dustActiveSourceMode();
  dustMaskEditor.timelineId = '';
  dustMaskEditor.editRevision = 0;
  dustMaskEditor.zipSha256 = '';
  dustMaskEditor.editSignature = '';
  dustMaskEditor.compatible = true;
  dustMaskEditor.compatibilityReason = '';
  dustMaskEditor.editedFrames = [];
  dustMaskEditor.reviewed = false;
  dustMaskEditor.summary = null;
  dustMaskEditor.statusText = '';
  dustMaskEditor.statusTone = '';
}

function dustMaskEditorSyncFromState(stateObj) {
  if (!stateObj || dustMaskEditorIsBusy()) return;
  const srcMode = _dustActiveSourceMode();
  if (dustMaskEditor.analysisSource !== srcMode) {
    if (dustMaskEditor.loading || dustMaskEditor.bitmask !== null) return;
    dustMaskEditor.analysisSource = srcMode;
  }
  const info =
    srcMode === 'original'
      ? (stateObj?.dust_edits?.original || stateObj?.dust_matrix?.original?.dust_edits || null)
      : (stateObj?.dust_edits?.stage || stateObj?.dust_edits || stateObj?.dust_matrix?.dust_edits || null);
  if (!info || typeof info !== 'object') {
    dustMaskEditorSyncUi();
    return;
  }
  if (info.timeline_id !== undefined) dustMaskEditor.timelineId = String(info.timeline_id || '');
  if (info.edit_revision !== undefined) dustMaskEditor.editRevision = Number(info.edit_revision || 0);
  if (info.zip_sha256 !== undefined) dustMaskEditor.zipSha256 = String(info.zip_sha256 || '');
  if (info.edit_signature !== undefined) dustMaskEditor.editSignature = String(info.edit_signature || '');
  if (info.compatible !== undefined) dustMaskEditor.compatible = info.compatible !== false;
  if (info.compatibility_reason !== undefined) {
    dustMaskEditor.compatibilityReason = String(info.compatibility_reason || '');
  }
  if (Array.isArray(info.edited_frames)) dustMaskEditor.editedFrames = info.edited_frames.slice();
  if (info.summary !== undefined) dustMaskEditor.summary = info.summary;
  dustMaskEditorSyncUi();
}

function _pushDustUndoSnapshot() {
  if (!_isUint8Array(dustMaskEditor.bitmask)) return;
  dustMaskEditor.undoStack.push(new Uint8Array(dustMaskEditor.bitmask));
  if (dustMaskEditor.undoStack.length > DUST_EDITOR_MAX_UNDO) {
    dustMaskEditor.undoStack.shift();
  }
  dustMaskEditor.redoStack.length = 0;
}

function _repaintEditorCanvas(dirtyRect = null) {
  const fr = dustMaskEditor.frameRender;
  if (!fr || !fr.canvas || !fr.canvas.isConnected) return;
  const result = renderDustMaskOverlayCanvas(fr.canvas, {
    width: fr.width,
    height: fr.height,
    baseImage: fr.baseImage,
    mode: fr.mode,
    bitmask: dustMaskEditor.bitmask,
    autoScratch: fr.autoScratch,
    autoDust: fr.autoDust,
    autoUncertain: fr.autoUncertain,
    autoVetoed: fr.autoVetoed,
    autoCombined: fr.autoCombined,
    opacity: fr.opacity,
    layers: dustMaskEditor.layers,
    splitActive: fr.splitActive,
    splitPos: fr.splitPos,
    overlayCanvas: dustMaskEditor.overlayCanvas,
    overlayImageData: dustMaskEditor.overlayImageData,
    dirtyRect,
  });
  if (result) {
    dustMaskEditor.overlayCanvas = result.overlayCanvas;
    dustMaskEditor.overlayImageData = result.overlayImageData;
  }
}

function undoDustMaskEdit() {
  if (!dustMaskEditor.compatible || !_isUint8Array(dustMaskEditor.bitmask)) return false;
  if (dustMaskEditor.undoStack.length === 0) return false;
  dustMaskEditor.redoStack.push(new Uint8Array(dustMaskEditor.bitmask));
  if (dustMaskEditor.redoStack.length > DUST_EDITOR_MAX_UNDO) {
    dustMaskEditor.redoStack.shift();
  }
  dustMaskEditor.bitmask = dustMaskEditor.undoStack.pop();
  dustMaskEditor.dirty = true;
  dustMaskEditor.frameStats = dustComputeFrameStats(dustMaskEditor.bitmask);
  _repaintEditorCanvas();
  _scheduleDustMaskAutoSave();
  _dustSetStatus(`Undid stroke on frame ${dustMaskEditor.frame + 1}. Unsaved changes…`);
  return true;
}

function redoDustMaskEdit() {
  if (!dustMaskEditor.compatible || !_isUint8Array(dustMaskEditor.bitmask)) return false;
  if (dustMaskEditor.redoStack.length === 0) return false;
  dustMaskEditor.undoStack.push(new Uint8Array(dustMaskEditor.bitmask));
  if (dustMaskEditor.undoStack.length > DUST_EDITOR_MAX_UNDO) {
    dustMaskEditor.undoStack.shift();
  }
  dustMaskEditor.bitmask = dustMaskEditor.redoStack.pop();
  dustMaskEditor.dirty = true;
  dustMaskEditor.frameStats = dustComputeFrameStats(dustMaskEditor.bitmask);
  _repaintEditorCanvas();
  _scheduleDustMaskAutoSave();
  _dustSetStatus(`Redid stroke on frame ${dustMaskEditor.frame + 1}. Unsaved changes…`);
  return true;
}

function dustMaskEditorSetTool(tool) {
  if (!DUST_EDIT_TOOLS.includes(tool)) return;
  dustMaskEditor.tool = tool;
  dustMaskEditorSyncUi();
}

function dustMaskEditorSetBrushRadius(radius) {
  const r = Math.max(0, Math.min(30, Math.round(Number(radius) || 0)));
  dustMaskEditor.brushRadius = r;
  dustMaskEditorSyncUi();
}

function dustMaskEditorSetLayer(layer, visible) {
  if (!Object.prototype.hasOwnProperty.call(dustMaskEditor.layers, layer)) return;
  dustMaskEditor.layers[layer] = Boolean(visible);
  dustMaskEditor.overlayImageData = null;
  _repaintEditorCanvas();
  dustMaskEditorSyncUi();
}

function dustMaskEditorSetEditMode(enabled) {
  const next = Boolean(enabled);
  if (next === dustMaskEditor.editMode) return;
  if (!next) {
    dustMaskEditor.spacePressed = false;
    void saveCurrentDustMaskEdit();
  }
  dustMaskEditor.editMode = next;
  if (typeof dustInputMode !== 'undefined' && next && dustInputMode === 'input') {
    dustInputMode = 'overlay';
  }
  if (next && typeof denoiseWorkspaceSetExact === 'function') {
    denoiseWorkspaceSetExact(true);
  }
  if (typeof dustSetView === 'function') {
    dustSetView('mask');
  }
  if (dustMaskEditor.frameRender?.canvas) {
    const canvas = dustMaskEditor.frameRender.canvas;
    const splitOn = typeof dustAbSplitActive !== 'undefined' && dustAbSplitActive;
    canvas.style.cursor = next ? 'crosshair' : splitOn ? 'ew-resize' : 'default';
    canvas.style.pointerEvents = next || splitOn ? 'auto' : '';
    _repaintEditorCanvas();
  }
  dustMaskEditorSyncUi();
}

function _scheduleDustMaskAutoSave() {
  if (dustMaskEditor.saveTimer) clearTimeout(dustMaskEditor.saveTimer);
  dustMaskEditor.saveTimer = setTimeout(() => {
    dustMaskEditor.saveTimer = null;
    void saveCurrentDustMaskEdit();
  }, DUST_EDITOR_AUTOSAVE_DEBOUNCE_MS);
}

function saveCurrentDustMaskEdit() {
  if (dustMaskEditor.saveTimer) {
    clearTimeout(dustMaskEditor.saveTimer);
    dustMaskEditor.saveTimer = null;
  }
  if (
    !dustMaskEditor.dirty ||
    dustMaskEditor.frame < 0 ||
    !_isUint8Array(dustMaskEditor.bitmask) ||
    !dustMaskEditor.compatible
  ) {
    return dustMaskEditor.savePromise;
  }

  const frameToSave = dustMaskEditor.frame;
  const sourceToSave = dustMaskEditor.analysisSource;
  const maskCopy = new Uint8Array(dustMaskEditor.bitmask);
  const b64 = dustBitmaskToBase64(maskCopy);
  const projectPath = _dustProjectPath();
  dustMaskEditor.dirty = false;
  dustMaskEditor.saving = true;
  _dustSetStatus(`Saving edits for frame ${frameToSave + 1}…`);

  dustMaskEditor.savePromise = dustMaskEditor.savePromise
    .catch(() => {})
    .then(async () => {
      try {
        const res = await postJson('/api/dust-mask-frame', {
          frame: frameToSave,
          analysis_source: sourceToSave,
          edit_mask_b64: b64,
          expected_revision: dustMaskEditor.editRevision,
          timeline_id: dustMaskEditor.timelineId,
          project: projectPath,
        });
        if (!res || !res.ok) {
          throw new Error(res?.error || 'Could not save frame mask edits.');
        }
        const stillSameContext =
          sourceToSave === _dustActiveSourceMode() && projectPath === _dustProjectPath();
        if (stillSameContext) {
          dustMaskEditor.timelineId = String(res.timeline_id || dustMaskEditor.timelineId || '');
          dustMaskEditor.editRevision = Number(res.edit_revision ?? dustMaskEditor.editRevision);
          dustMaskEditor.zipSha256 = String(res.zip_sha256 || '');
          dustMaskEditor.editSignature = String(res.edit_signature || '');
          dustMaskEditor.editedFrames = Array.isArray(res.edited_frames) ? res.edited_frames.slice() : [];
          if (res.summary !== undefined) dustMaskEditor.summary = res.summary;
          if (dustMaskEditor.frame === frameToSave) {
            dustMaskEditor.reviewed = Boolean(res.reviewed);
            if (res.frame_stats) dustMaskEditor.frameStats = res.frame_stats;
          }
          if (res.state && typeof state !== 'undefined') {
            state = res.state;
          }
          if (!dustMaskEditor.dirty) {
            _dustSetStatus(
              `Saved frame ${frameToSave + 1} (rev ${dustMaskEditor.editRevision} · ${dustMaskEditor.editedFrames.length} edited frame${dustMaskEditor.editedFrames.length === 1 ? '' : 's'}).`,
              'ok'
            );
          }
        }
      } catch (err) {
        const stillSameContext =
          sourceToSave === _dustActiveSourceMode() && projectPath === _dustProjectPath();
        if (stillSameContext && dustMaskEditor.frame === frameToSave && !dustMaskEditor.dirty) {
          dustMaskEditor.dirty = true;
        }
        if (stillSameContext) {
          _dustSetStatus(`Save failed on frame ${frameToSave + 1}: ${err?.message || err}`, 'error');
        }
      } finally {
        dustMaskEditor.saving = false;
        dustMaskEditorSyncUi();
      }
    });

  return dustMaskEditor.savePromise;
}

function dustMaskEditorFlushSave() {
  return saveCurrentDustMaskEdit();
}

async function dustMaskEditorLoadFrame(frame, analysisSource = _dustActiveSourceMode()) {
  const targetFrame = Math.max(0, Math.round(Number(frame) || 0));
  const sourceMode = analysisSource === 'original' ? 'original' : 'stage';

  if (
    dustMaskEditor.dirty &&
    (dustMaskEditor.frame !== targetFrame || dustMaskEditor.analysisSource !== sourceMode)
  ) {
    saveCurrentDustMaskEdit();
  }

  const ticket = ++dustMaskEditor.loadEpoch;
  dustMaskEditor.loading = true;
  dustMaskEditor.loadingFrame = targetFrame;
  dustMaskEditor.frame = -1;
  dustMaskEditor.analysisSource = sourceMode;
  dustMaskEditor.bitmask = null;
  dustMaskEditor.undoStack = [];
  dustMaskEditor.redoStack = [];
  dustMaskEditor.dirty = false;
  dustMaskEditor.reviewed = false;
  dustMaskEditor.frameStats = null;
  dustMaskEditor.overlayImageData = null;
  _repaintEditorCanvas();
  dustMaskEditorSyncUi();

  try {
    await dustMaskEditor.savePromise;
  } catch (_err) {
    // Error already surfaced via status.
  }
  if (ticket !== dustMaskEditor.loadEpoch) return null;

  try {
    const qs = new URLSearchParams({
      frame: String(targetFrame),
      analysis_source: sourceMode,
    });
    const res = await api('/api/dust-mask-frame?' + qs.toString());
    if (ticket !== dustMaskEditor.loadEpoch) return null;
    if (!res || !res.ok) {
      throw new Error(res?.error || 'Could not load frame edit bitmask.');
    }

    const w = Number(res.width || 0);
    const h = Number(res.height || 0);
    dustMaskEditor.loading = false;
    dustMaskEditor.loadingFrame = -1;
    dustMaskEditor.frame = Number(res.frame ?? targetFrame);
    dustMaskEditor.frameCount = Number(res.frame_count || 0);
    dustMaskEditor.width = w;
    dustMaskEditor.height = h;
    dustMaskEditor.analysisSource = res.analysis_source || sourceMode;
    dustMaskEditor.timelineId = String(res.timeline_id || '');
    dustMaskEditor.editRevision = Number(res.edit_revision || 0);
    dustMaskEditor.zipSha256 = String(res.zip_sha256 || '');
    dustMaskEditor.editSignature = String(res.edit_signature || '');
    dustMaskEditor.compatible = res.compatible !== false;
    dustMaskEditor.compatibilityReason = String(res.compatibility_reason || '');
    dustMaskEditor.editedFrames = Array.isArray(res.edited_frames) ? res.edited_frames.slice() : [];
    dustMaskEditor.reviewed = Boolean(res.reviewed);
    dustMaskEditor.frameStats = res.frame_stats || null;
    dustMaskEditor.summary = res.summary || null;

    if (w > 0 && h > 0) {
      dustMaskEditor.bitmask = dustBase64ToBitmask(res.edit_mask_b64 || '', w, h);
      if (!dustMaskEditor.frameStats) {
        dustMaskEditor.frameStats = dustComputeFrameStats(dustMaskEditor.bitmask);
      }
    } else {
      dustMaskEditor.bitmask = null;
    }

    dustMaskEditor.undoStack = [];
    dustMaskEditor.redoStack = [];
    dustMaskEditor.dirty = false;
    dustMaskEditor.overlayImageData = null;
    _repaintEditorCanvas();

    if (!dustMaskEditor.compatible) {
      _dustSetStatus(
        `Timeline mismatch: ${dustMaskEditor.compatibilityReason || 'Edit archive incompatible.'}`,
        'warn'
      );
    } else {
      _dustSetStatus(
        dustMaskEditor.editedFrames.includes(dustMaskEditor.frame)
          ? `Frame ${dustMaskEditor.frame + 1} loaded (has human edits · rev ${dustMaskEditor.editRevision}).`
          : `Frame ${dustMaskEditor.frame + 1} loaded (no manual edits yet).`
      );
    }
    return res;
  } catch (err) {
    if (ticket !== dustMaskEditor.loadEpoch) return null;
    dustMaskEditor.loading = false;
    dustMaskEditor.loadingFrame = -1;
    _dustSetStatus(`Mask edit load failed: ${err?.message || err}`, 'error');
    return null;
  }
}

function dustMaskEditorAttachFrameInspection(canvas, frameParams) {
  if (!canvas || !frameParams) return;
  const fr = {
    canvas,
    frame: Number(frameParams.frame),
    width: Number(frameParams.width),
    height: Number(frameParams.height),
    baseImage: frameParams.baseImage || null,
    mode: frameParams.mode || 'overlay',
    autoScratch: frameParams.autoScratch || null,
    autoDust: frameParams.autoDust || null,
    autoUncertain: frameParams.autoUncertain || null,
    autoVetoed: frameParams.autoVetoed || null,
    autoCombined: frameParams.autoCombined || null,
    opacity: Number(frameParams.opacity ?? 0.55),
    splitActive: Boolean(frameParams.splitActive),
    splitPos: Number(frameParams.splitPos ?? 50),
  };

  const activeSource = _dustActiveSourceMode();
  const sameFrame =
    !dustMaskEditor.loading &&
    dustMaskEditor.frame === fr.frame &&
    dustMaskEditor.analysisSource === activeSource &&
    dustMaskEditor.width === fr.width &&
    dustMaskEditor.height === fr.height &&
    _isUint8Array(dustMaskEditor.bitmask) &&
    dustMaskEditor.bitmask.length === fr.width * fr.height;
  if (!sameFrame && !dustMaskEditor.loading) {
    dustMaskEditor.frame = fr.frame;
    dustMaskEditor.analysisSource = activeSource;
    dustMaskEditor.width = fr.width;
    dustMaskEditor.height = fr.height;
    dustMaskEditor.bitmask = new Uint8Array(fr.width * fr.height);
  }
  dustMaskEditor.frameRender = fr;
  dustMaskEditor.overlayImageData = null;
  _repaintEditorCanvas();
  dustMaskEditorSyncUi();
}

function _applyBrushAtClientPoint(canvas, clientX, clientY, isStart) {
  if (!dustMaskEditor.compatible) {
    _dustSetStatus('Cannot paint while edit archive has a timeline mismatch. Use Reset All to clear incompatible edits.', 'warn');
    return false;
  }
  if (
    dustMaskEditor.loading ||
    !dustMaskEditor.frameRender ||
    dustMaskEditor.frame !== dustMaskEditor.frameRender.frame
  ) {
    _dustSetStatus('Loading frame mask edits… please wait a moment.', 'warn');
    return false;
  }
  if (!_isUint8Array(dustMaskEditor.bitmask) || !dustMaskEditor.width || !dustMaskEditor.height) {
    return false;
  }
  const pt = dustCanvasClientToPixel(canvas, clientX, clientY);
  if (!pt) return false;
  if (isStart) {
    _pushDustUndoSnapshot();
  }
  const prev = dustMaskEditor.lastPoint || pt;
  const opts = { inPlace: true };
  paintBrushSegment(
    dustMaskEditor.bitmask,
    dustMaskEditor.width,
    dustMaskEditor.height,
    prev.x,
    prev.y,
    pt.x,
    pt.y,
    dustMaskEditor.brushRadius,
    dustMaskEditor.tool,
    opts
  );
  dustMaskEditor.lastPoint = pt;
  dustMaskEditor.dirty = true;

  let dirtyRect = opts.lastDirtyRect;
  if (dirtyRect) {
    dirtyRect = {
      minX: Math.max(0, dirtyRect.minX - 1),
      minY: Math.max(0, dirtyRect.minY - 1),
      maxX: Math.min(dustMaskEditor.width - 1, dirtyRect.maxX + 1),
      maxY: Math.min(dustMaskEditor.height - 1, dirtyRect.maxY + 1),
    };
  }
  _repaintEditorCanvas(dirtyRect);
  return true;
}

function _finishStroke(canvas) {
  if (!dustMaskEditor.drawing) return;
  dustMaskEditor.drawing = false;
  dustMaskEditor.lastPoint = null;
  const pid = dustMaskEditor.activePointerId;
  dustMaskEditor.activePointerId = null;
  if (pid !== null && canvas && canvas.hasPointerCapture?.(pid)) {
    try {
      canvas.releasePointerCapture(pid);
    } catch (_err) {}
  }
  if (_isUint8Array(dustMaskEditor.bitmask)) {
    dustMaskEditor.frameStats = dustComputeFrameStats(dustMaskEditor.bitmask);
  }
  if (dustMaskEditor.dirty) {
    _dustSetStatus(`Frame ${dustMaskEditor.frame + 1} edited. Auto-saving…`);
    _scheduleDustMaskAutoSave();
  }
  dustMaskEditorSyncUi();
}

function dustMaskEditorHandlePointerDown(canvas, event) {
  if (!dustMaskEditor.editMode) return false;
  if (dustMaskEditor.spacePressed || (event.pointerType === 'mouse' && event.button !== 0)) {
    return false;
  }
  event.stopPropagation();
  event.preventDefault();

  if (dustMaskEditor.saveTimer) {
    clearTimeout(dustMaskEditor.saveTimer);
    dustMaskEditor.saveTimer = null;
  }

  dustMaskEditor.lastPoint = null;
  const started = _applyBrushAtClientPoint(canvas, event.clientX, event.clientY, true);
  if (!started) return true;

  dustMaskEditor.drawing = true;
  dustMaskEditor.activePointerId = event.pointerId ?? null;
  try {
    if (event.pointerId != null) canvas.setPointerCapture?.(event.pointerId);
  } catch (_err) {}

  const onMove = moveEvt => {
    if (!dustMaskEditor.drawing) return;
    if (dustMaskEditor.activePointerId !== null && moveEvt.pointerId !== dustMaskEditor.activePointerId) return;
    _applyBrushAtClientPoint(canvas, moveEvt.clientX, moveEvt.clientY, false);
  };
  const onUp = upEvt => {
    if (dustMaskEditor.activePointerId !== null && upEvt.pointerId != null && upEvt.pointerId !== dustMaskEditor.activePointerId) {
      return;
    }
    cleanup();
    _finishStroke(canvas);
  };
  const cleanup = () => {
    if (typeof window !== 'undefined') {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      window.removeEventListener('blur', onUp);
    }
  };
  if (typeof window !== 'undefined') {
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    window.addEventListener('blur', onUp);
  }
  return true;
}

function dustMaskEditorHandleMouseDown(canvas, event) {
  if (!dustMaskEditor.editMode) return false;
  if (dustMaskEditor.drawing) return true;
  if (dustMaskEditor.spacePressed || (event.button !== undefined && event.button !== 0)) {
    return false;
  }
  event.stopPropagation?.();
  event.preventDefault?.();

  if (dustMaskEditor.saveTimer) {
    clearTimeout(dustMaskEditor.saveTimer);
    dustMaskEditor.saveTimer = null;
  }
  dustMaskEditor.lastPoint = null;
  const started = _applyBrushAtClientPoint(canvas, event.clientX, event.clientY, true);
  if (!started) return true;
  dustMaskEditor.drawing = true;

  const onMove = moveEvt => {
    if (!dustMaskEditor.drawing) return;
    _applyBrushAtClientPoint(canvas, moveEvt.clientX, moveEvt.clientY, false);
  };
  const onUp = () => {
    cleanup();
    _finishStroke(canvas);
  };
  const cleanup = () => {
    if (typeof window !== 'undefined') {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    }
  };
  if (typeof window !== 'undefined') {
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }
  return true;
}

async function goToDustFrame(frame) {
  const num = Number(frame);
  if (!Number.isFinite(num)) return;
  const maxIdx = Math.max(0, (dustMaskEditor.frameCount || 1) - 1);
  const clamped = Math.max(0, Math.min(maxIdx, Math.round(num)));
  await saveCurrentDustMaskEdit();
  if (typeof denoiseWorkspaceFor === 'function') {
    denoiseWorkspaceFor().seekFrame(clamped);
  }
}

function _sortedEditedFramesWithCurrent() {
  const set = new Set(dustMaskEditor.editedFrames || []);
  if (dustMaskEditor.frame >= 0 && dustBitmaskHasEdits(dustMaskEditor.bitmask)) {
    set.add(dustMaskEditor.frame);
  }
  return [...set].sort((a, b) => a - b);
}

function prevEditedFrame() {
  const cur = dustMaskEditor.frame >= 0
    ? dustMaskEditor.frame
    : typeof denoiseWorkspaceSnapshot === 'function'
      ? Number(denoiseWorkspaceSnapshot().frame || 0)
      : 0;
  const frames = _sortedEditedFramesWithCurrent();
  const prev = [...frames].reverse().find(f => f < cur);
  if (prev !== undefined) void goToDustFrame(prev);
}

function nextEditedFrame() {
  const cur = dustMaskEditor.frame >= 0
    ? dustMaskEditor.frame
    : typeof denoiseWorkspaceSnapshot === 'function'
      ? Number(denoiseWorkspaceSnapshot().frame || 0)
      : 0;
  const frames = _sortedEditedFramesWithCurrent();
  const next = frames.find(f => f > cur);
  if (next !== undefined) void goToDustFrame(next);
}

async function markCurrentDustMaskReviewed() {
  if (!dustMaskEditor.compatible || !_isUint8Array(dustMaskEditor.bitmask)) return;
  _pushDustUndoSnapshot();
  const allIndices = new Uint8Array(dustMaskEditor.bitmask.length);
  allIndices.fill(1);
  dustApplyStrokeTransition(dustMaskEditor.bitmask, allIndices, 'mark_reviewed', { inPlace: true });
  dustMaskEditor.dirty = true;
  dustMaskEditor.reviewed = true;
  dustMaskEditor.frameStats = dustComputeFrameStats(dustMaskEditor.bitmask);
  _repaintEditorCanvas();
  await saveCurrentDustMaskEdit();
}

async function resetCurrentDustMaskFrame() {
  if (!dustMaskEditor.compatible || dustMaskEditor.frame < 0) return;
  if (dustMaskEditor.saveTimer) {
    clearTimeout(dustMaskEditor.saveTimer);
    dustMaskEditor.saveTimer = null;
  }
  const targetFrame = dustMaskEditor.frame;
  const sourceMode = _dustActiveSourceMode();
  const projectPath = _dustProjectPath();

  try {
    await dustMaskEditor.savePromise;
    const res = await postJson('/api/dust-mask-reset', {
      scope: 'frame',
      frame: targetFrame,
      analysis_source: sourceMode,
      timeline_id: dustMaskEditor.timelineId,
      project: projectPath,
    });
    if (!res || !res.ok) throw new Error(res?.error || 'Could not reset frame edits.');
    if (sourceMode !== _dustActiveSourceMode() || projectPath !== _dustProjectPath()) return;
    if (dustMaskEditor.frame === targetFrame && _isUint8Array(dustMaskEditor.bitmask)) {
      if (dustBitmaskHasEdits(dustMaskEditor.bitmask)) {
        _pushDustUndoSnapshot();
        dustMaskEditor.bitmask.fill(0);
      }
      dustMaskEditor.dirty = false;
      dustMaskEditor.reviewed = false;
      dustMaskEditor.frameStats = dustComputeFrameStats(dustMaskEditor.bitmask);
    }
    dustMaskEditor.timelineId = String(res.timeline_id || dustMaskEditor.timelineId || '');
    dustMaskEditor.editRevision = Number(res.edit_revision || 0);
    dustMaskEditor.zipSha256 = String(res.zip_sha256 || '');
    dustMaskEditor.editSignature = String(res.edit_signature || '');
    dustMaskEditor.editedFrames = Array.isArray(res.edited_frames) ? res.edited_frames.slice() : [];
    dustMaskEditor.summary = res.summary || null;
    if (res.state && typeof state !== 'undefined') state = res.state;
    _repaintEditorCanvas();
    _dustSetStatus(`Reset edits on frame ${targetFrame + 1} (Ctrl+Z to undo).`, 'ok');
  } catch (err) {
    _dustSetStatus(`Reset frame failed: ${err?.message || err}`, 'error');
  }
}

async function resetAllDustMaskEdits() {
  const count = (dustMaskEditor.editedFrames || []).length;
  const msg = count
    ? `Clear manual mask edits across all ${count} edited frame${count === 1 ? '' : 's'}? Automatic detector caches are preserved. This cannot be undone.`
    : 'Clear all saved manual mask edits for this timeline? Automatic detector caches are preserved.';
  if (typeof confirm === 'function' && !confirm(msg)) return;

  if (dustMaskEditor.saveTimer) {
    clearTimeout(dustMaskEditor.saveTimer);
    dustMaskEditor.saveTimer = null;
  }
  const sourceMode = _dustActiveSourceMode();
  const projectPath = _dustProjectPath();
  try {
    await dustMaskEditor.savePromise;
    const res = await postJson('/api/dust-mask-reset', {
      scope: 'all',
      frame: Math.max(0, dustMaskEditor.frame),
      analysis_source: sourceMode,
      project: projectPath,
    });
    if (!res || !res.ok) throw new Error(res?.error || 'Could not reset all mask edits.');
    if (sourceMode !== _dustActiveSourceMode() || projectPath !== _dustProjectPath()) return;
    if (_isUint8Array(dustMaskEditor.bitmask)) {
      dustMaskEditor.bitmask.fill(0);
    }
    dustMaskEditor.dirty = false;
    dustMaskEditor.undoStack = [];
    dustMaskEditor.redoStack = [];
    dustMaskEditor.compatible = true;
    dustMaskEditor.compatibilityReason = '';
    dustMaskEditor.timelineId = String(res.timeline_id || dustMaskEditor.timelineId || '');
    dustMaskEditor.editRevision = Number(res.edit_revision || 0);
    dustMaskEditor.zipSha256 = String(res.zip_sha256 || '');
    dustMaskEditor.editSignature = String(res.edit_signature || '');
    dustMaskEditor.editedFrames = Array.isArray(res.edited_frames) ? res.edited_frames.slice() : [];
    dustMaskEditor.reviewed = false;
    dustMaskEditor.frameStats = dustComputeFrameStats(dustMaskEditor.bitmask);
    dustMaskEditor.summary = res.summary || null;
    if (res.state && typeof state !== 'undefined') state = res.state;
    _repaintEditorCanvas();
    _dustSetStatus('Cleared all manual mask edits for this timeline.', 'ok');
  } catch (err) {
    _dustSetStatus(`Reset all failed: ${err?.message || err}`, 'error');
  }
}

async function exportDustMaskProvenance(values) {
  if (dustMaskEditor.exportPending) return;
  dustMaskEditor.exportPending = true;
  _dustSetStatus('Exporting mask provenance bundle…');
  try {
    await saveCurrentDustMaskEdit();
    const draftValues =
      values || (typeof dustDraft !== 'undefined' && dustDraft ? { ...dustDraft } : {});
    const res = await postJson('/api/dust-mask-export', {
      analysis_source: _dustActiveSourceMode(),
      values: draftValues,
      include_auto_proposals: true,
      project: _dustProjectPath(),
    });
    if (!res || !res.ok) throw new Error(res?.error || 'Provenance export failed.');
    const count = Array.isArray(res.exported_frames) ? res.exported_frames.length : 0;
    const loc = res.provenance_path || res.export_dir || 'export directory';
    _dustSetStatus(
      `Exported operational provenance (${count} frame${count === 1 ? '' : 's'}) to ${loc}. Operational edits only, not independent GT.`,
      'ok'
    );
  } catch (err) {
    _dustSetStatus(`Export failed: ${err?.message || err}`, 'error');
  } finally {
    dustMaskEditor.exportPending = false;
    dustMaskEditorSyncUi();
  }
}

function dustMaskEditorInstallKeybindings() {
  if (dustMaskEditor.keybindingsInstalled || typeof document === 'undefined') return;
  dustMaskEditor.keybindingsInstalled = true;

  document.addEventListener('keydown', event => {
    if (typeof active === 'string' && active !== 'dust') return;
    if (!dustMaskEditor.editMode) return;
    const target = event.target;
    const tag = target?.tagName;
    if (
      tag === 'TEXTAREA' ||
      tag === 'SELECT' ||
      (tag === 'INPUT' && !['checkbox', 'button', 'range'].includes(target.type))
    ) {
      return;
    }

    if (event.code === 'Space') {
      if (target === document.body || target?.closest?.('.dust-mask-review')) {
        dustMaskEditor.spacePressed = true;
        if (dustMaskEditor.frameRender?.canvas) {
          dustMaskEditor.frameRender.canvas.style.cursor = 'grab';
        }
        event.preventDefault();
      }
      return;
    }

    const key = String(event.key || '').toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === 'z') {
      event.preventDefault();
      if (event.shiftKey) redoDustMaskEdit();
      else undoDustMaskEdit();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && key === 'y') {
      event.preventDefault();
      redoDustMaskEdit();
      return;
    }
    if (event.key === '[') {
      event.preventDefault();
      dustMaskEditorSetBrushRadius(dustMaskEditor.brushRadius - 1);
      return;
    }
    if (event.key === ']') {
      event.preventDefault();
      dustMaskEditorSetBrushRadius(dustMaskEditor.brushRadius + 1);
    }
  });

  document.addEventListener('keyup', event => {
    if (event.code === 'Space' && dustMaskEditor.spacePressed) {
      dustMaskEditor.spacePressed = false;
      if (dustMaskEditor.frameRender?.canvas && dustMaskEditor.editMode) {
        dustMaskEditor.frameRender.canvas.style.cursor = 'crosshair';
      }
    }
  });

  if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
    window.addEventListener('blur', () => {
      if (dustMaskEditor.spacePressed) {
        dustMaskEditor.spacePressed = false;
        if (dustMaskEditor.frameRender?.canvas && dustMaskEditor.editMode) {
          dustMaskEditor.frameRender.canvas.style.cursor = 'crosshair';
        }
      }
    });
  }
}

function dustMaskEditorMountUi(viewerEl) {
  if (!viewerEl || typeof document === 'undefined') return;
  dustMaskEditorInstallKeybindings();

  const toggle = viewerEl.querySelector('[data-dust-edit-toggle]');
  if (toggle) {
    toggle.onclick = () => dustMaskEditorSetEditMode(!dustMaskEditor.editMode);
  }

  viewerEl.querySelectorAll('[data-dust-tool]').forEach(btn => {
    btn.onclick = () => dustMaskEditorSetTool(btn.dataset.dustTool);
  });

  const radiusInput = viewerEl.querySelector('[data-dust-brush-radius]');
  if (radiusInput) {
    radiusInput.oninput = e => dustMaskEditorSetBrushRadius(Number(e.target.value));
  }

  viewerEl.querySelectorAll('[data-dust-layer]').forEach(chk => {
    chk.onchange = e => dustMaskEditorSetLayer(chk.dataset.dustLayer, e.target.checked);
  });

  const undoBtn = viewerEl.querySelector('[data-dust-undo]');
  if (undoBtn) undoBtn.onclick = () => undoDustMaskEdit();
  const redoBtn = viewerEl.querySelector('[data-dust-redo]');
  if (redoBtn) redoBtn.onclick = () => redoDustMaskEdit();
  const saveBtn = viewerEl.querySelector('[data-dust-save]');
  if (saveBtn) saveBtn.onclick = () => void saveCurrentDustMaskEdit();
  const reviewBtn = viewerEl.querySelector('[data-dust-mark-reviewed]');
  if (reviewBtn) reviewBtn.onclick = () => void markCurrentDustMaskReviewed();
  const resetFrameBtn = viewerEl.querySelector('[data-dust-reset-frame]');
  if (resetFrameBtn) resetFrameBtn.onclick = () => void resetCurrentDustMaskFrame();
  const resetAllBtn = viewerEl.querySelector('[data-dust-reset-all]');
  if (resetAllBtn) resetAllBtn.onclick = () => void resetAllDustMaskEdits();
  const compatResetBtn = viewerEl.querySelector('[data-dust-compat-reset-all]');
  if (compatResetBtn) compatResetBtn.onclick = () => void resetAllDustMaskEdits();

  const prevBtn = viewerEl.querySelector('[data-dust-prev-edited]');
  if (prevBtn) prevBtn.onclick = () => prevEditedFrame();
  const nextBtn = viewerEl.querySelector('[data-dust-next-edited]');
  if (nextBtn) nextBtn.onclick = () => nextEditedFrame();
  const exportBtn = viewerEl.querySelector('[data-dust-export-provenance]');
  if (exportBtn) exportBtn.onclick = () => void exportDustMaskProvenance();

  dustMaskEditorSyncUi();
}

function dustMaskEditorSyncUi() {
  if (typeof document === 'undefined' || typeof document.querySelector !== 'function') return;
  const viewer = document.querySelector('.dust-mask-review');
  if (!viewer) return;

  const toggle = viewer.querySelector('[data-dust-edit-toggle]');
  if (toggle) {
    toggle.setAttribute('aria-pressed', String(dustMaskEditor.editMode));
    toggle.classList.toggle('is-active', dustMaskEditor.editMode);
    toggle.textContent = dustMaskEditor.editMode ? 'Edit Mask: ON' : 'Edit Mask';
  }

  const bar = viewer.querySelector('[data-dust-editor-bar]');
  if (bar) bar.hidden = !dustMaskEditor.editMode;

  const compatBanner = viewer.querySelector('[data-dust-compat-banner]');
  if (compatBanner) {
    compatBanner.hidden = dustMaskEditor.compatible;
    if (!dustMaskEditor.compatible) {
      const msg =
        `Edit archive mismatch (${dustMaskEditor.compatibilityReason || 'source/geometry changed'}). ` +
        'Editing is locked to prevent silent misalignment. Use Reset All to clear incompatible edits.';
      const msgEl = compatBanner.querySelector('[data-dust-compat-text]');
      if (msgEl) msgEl.textContent = msg;
      else compatBanner.textContent = msg;
    }
  }

  viewer.querySelectorAll('[data-dust-tool]').forEach(btn => {
    const activeTool = btn.dataset.dustTool === dustMaskEditor.tool;
    btn.setAttribute('aria-pressed', String(activeTool));
    btn.classList.toggle('is-active', activeTool);
    btn.disabled = !dustMaskEditor.compatible;
  });

  const radiusInput = viewer.querySelector('[data-dust-brush-radius]');
  if (radiusInput) radiusInput.value = String(dustMaskEditor.brushRadius);
  const radiusOut = viewer.querySelector('[data-dust-brush-readout]');
  if (radiusOut) {
    const r = dustMaskEditor.brushRadius;
    radiusOut.textContent = r === 0 ? '1 px (point)' : `${r * 2 + 1} px (r=${r})`;
  }

  const isE2 =
    (typeof dustDraft !== 'undefined' && dustDraft?.detector === 'experimental_e2') ||
    Boolean(dustMaskEditor.frameRender?.autoVetoed);
  viewer.querySelectorAll('[data-dust-layer]').forEach(chk => {
    const key = chk.dataset.dustLayer;
    if (Object.prototype.hasOwnProperty.call(dustMaskEditor.layers, key)) {
      chk.checked = Boolean(dustMaskEditor.layers[key]);
    }
    if (key === 'vetoed') {
      const wrap = chk.closest('[data-dust-layer-wrap="vetoed"]');
      if (wrap) wrap.hidden = !isE2;
    }
  });
  const vetoLegend = viewer.querySelector('[data-dust-legend-vetoed]');
  if (vetoLegend) vetoLegend.hidden = !isE2;

  const undoBtn = viewer.querySelector('[data-dust-undo]');
  if (undoBtn) undoBtn.disabled = !dustMaskEditor.compatible || dustMaskEditor.undoStack.length === 0;
  const redoBtn = viewer.querySelector('[data-dust-redo]');
  if (redoBtn) redoBtn.disabled = !dustMaskEditor.compatible || dustMaskEditor.redoStack.length === 0;
  const saveBtn = viewer.querySelector('[data-dust-save]');
  if (saveBtn) {
    saveBtn.disabled = !dustMaskEditor.compatible || !dustMaskEditor.dirty || dustMaskEditor.saving;
    saveBtn.textContent = dustMaskEditor.saving ? 'Saving…' : dustMaskEditor.dirty ? 'Save*' : 'Saved';
  }
  const reviewBtn = viewer.querySelector('[data-dust-mark-reviewed]');
  if (reviewBtn) {
    reviewBtn.disabled = !dustMaskEditor.compatible || !_isUint8Array(dustMaskEditor.bitmask);
  }
  const resetFrameBtn = viewer.querySelector('[data-dust-reset-frame]');
  if (resetFrameBtn) {
    resetFrameBtn.disabled = !dustMaskEditor.compatible || dustMaskEditor.frame < 0;
  }

  const edited = _sortedEditedFramesWithCurrent();
  const curFrame = dustMaskEditor.frame >= 0
    ? dustMaskEditor.frame
    : typeof denoiseWorkspaceSnapshot === 'function'
      ? Number(denoiseWorkspaceSnapshot().frame || 0)
      : 0;
  const curEdited = edited.includes(curFrame);
  const badge = viewer.querySelector('[data-dust-edited-badge]');
  if (badge) {
    const revPart = dustMaskEditor.editRevision > 0 ? ` · rev ${dustMaskEditor.editRevision}` : '';
    const frameTag = curEdited ? ' · current frame edited' : '';
    const reviewTag = dustMaskEditor.reviewed ? ' · reviewed' : '';
    badge.textContent = `Edited frames: ${edited.length}${revPart}${frameTag}${reviewTag}`;
    badge.classList.toggle('has-edits', edited.length > 0);
  }

  const prevBtn = viewer.querySelector('[data-dust-prev-edited]');
  if (prevBtn) prevBtn.disabled = !edited.some(f => f < curFrame);
  const nextBtn = viewer.querySelector('[data-dust-next-edited]');
  if (nextBtn) nextBtn.disabled = !edited.some(f => f > curFrame);

  const exportBtn = viewer.querySelector('[data-dust-export-provenance]');
  if (exportBtn) {
    exportBtn.disabled = dustMaskEditor.exportPending || !dustMaskEditor.compatible;
    exportBtn.textContent = dustMaskEditor.exportPending ? 'Exporting…' : 'Export Provenance…';
  }

  const statsEl = viewer.querySelector('[data-dust-frame-stats]');
  if (statsEl) {
    const st = dustMaskEditor.frameStats || dustComputeFrameStats(dustMaskEditor.bitmask);
    const parts = [
      `+Scratch: ${st.scratch_add_pixels}`,
      `+Dust: ${st.dust_add_pixels}`,
      `Rejected: ${st.scratch_reject_pixels + st.dust_reject_pixels + st.uncertain_reject_pixels}`,
      `Protect Intact: ${st.protect_intact_pixels}`,
    ];
    if (dustMaskEditor.reviewed) parts.push('Operational review marked');
    statsEl.textContent = parts.join(' · ');
  }

  const statusEl = viewer.querySelector('[data-dust-edit-status]');
  if (statusEl) {
    statusEl.textContent = dustMaskEditor.statusText;
    statusEl.dataset.tone = dustMaskEditor.statusTone || '';
  }
}

if (typeof globalThis !== 'undefined') {
  Object.assign(globalThis, {
    BIT_SCRATCH_ADD,
    BIT_SCRATCH_REJECT,
    BIT_DUST_ADD,
    BIT_DUST_REJECT,
    BIT_PROTECT_INTACT,
    BIT_UNCERTAIN_REJECT,
    BIT_REVIEWED,
    KNOWN_EDIT_BITS_MASK,
    DUST_EDIT_TOOLS,
    DUST_EDITOR_MAX_UNDO,
    dustBitmaskToBase64,
    dustBase64ToBitmask,
    dustBitmaskHasEdits,
    dustComputeFrameStats,
    dustApplyStrokeTransition,
    dustCanvasClientToPixel,
    dustClientPointToPixel,
    dustCollectBrushSegmentPixels,
    paintBrushSegment,
    dustExtractBinaryPlane,
    renderDustMaskOverlayCanvas,
    dustMaskEditor,
    dustMaskEditorIsUint8,
    dustMaskEditorIsBusy,
    dustMaskEditorIsEditActive,
    dustMaskEditorSetEditMode,
    dustMaskEditorSetTool,
    dustMaskEditorSetBrushRadius,
    dustMaskEditorSetLayer,
    dustMaskEditorSyncFromState,
    dustMaskEditorResetProject,
    dustMaskEditorResetLocalCache,
    dustMaskEditorLoadFrame,
    dustMaskEditorAttachFrameInspection,
    dustMaskEditorHandlePointerDown,
    dustMaskEditorHandleMouseDown,
    undoDustMaskEdit,
    redoDustMaskEdit,
    saveCurrentDustMaskEdit,
    dustMaskEditorFlushSave,
    goToDustFrame,
    prevEditedFrame,
    nextEditedFrame,
    markCurrentDustMaskReviewed,
    resetCurrentDustMaskFrame,
    resetAllDustMaskEdits,
    exportDustMaskProvenance,
    dustMaskEditorMountUi,
    dustMaskEditorSyncUi,
    dustMaskEditorInstallKeybindings,
  });
}
