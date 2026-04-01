import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

const iconSize = 64;

const colors = {
  transparent: [0, 0, 0, 0],
  compareBackground: [8, 11, 17, 255],
  compareDivider: [28, 35, 48, 255],
  badge: [9, 12, 18, 255],
  ring: [22, 28, 40, 255],
  mark: [178, 224, 255, 255],
};

function createCanvas(width, height, background = colors.transparent) {
  const pixels = new Uint8Array(width * height * 4);

  for (let index = 0; index < pixels.length; index += 4) {
    pixels[index] = background[0];
    pixels[index + 1] = background[1];
    pixels[index + 2] = background[2];
    pixels[index + 3] = background[3];
  }

  return { width, height, pixels };
}

function setPixel(canvas, x, y, color) {
  if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) {
    return;
  }

  const index = (y * canvas.width + x) * 4;
  canvas.pixels[index] = color[0];
  canvas.pixels[index + 1] = color[1];
  canvas.pixels[index + 2] = color[2];
  canvas.pixels[index + 3] = color[3];
}

function fillRect(canvas, startX, startY, width, height, color) {
  const endX = startX + width;
  const endY = startY + height;

  for (let y = startY; y < endY; y += 1) {
    for (let x = startX; x < endX; x += 1) {
      setPixel(canvas, x, y, color);
    }
  }
}

function fillCircle(canvas, centerX, centerY, radius, color) {
  const minX = Math.floor(centerX - radius);
  const maxX = Math.ceil(centerX + radius);
  const minY = Math.floor(centerY - radius);
  const maxY = Math.ceil(centerY + radius);
  const radiusSquared = radius * radius;

  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const dx = x + 0.5 - centerX;
      const dy = y + 0.5 - centerY;

      if (dx * dx + dy * dy <= radiusSquared) {
        setPixel(canvas, x, y, color);
      }
    }
  }
}

function fillSegment(canvas, x1, y1, x2, y2, thickness, color) {
  const minX = Math.floor(Math.min(x1, x2) - thickness);
  const maxX = Math.ceil(Math.max(x1, x2) + thickness);
  const minY = Math.floor(Math.min(y1, y2) - thickness);
  const maxY = Math.ceil(Math.max(y1, y2) + thickness);
  const radius = thickness / 2;
  const radiusSquared = radius * radius;
  const segmentX = x2 - x1;
  const segmentY = y2 - y1;
  const segmentLengthSquared = segmentX * segmentX + segmentY * segmentY;

  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const pointX = x + 0.5;
      const pointY = y + 0.5;

      let t = 0;

      if (segmentLengthSquared > 0) {
        t =
          ((pointX - x1) * segmentX + (pointY - y1) * segmentY) /
          segmentLengthSquared;
        t = Math.max(0, Math.min(1, t));
      }

      const nearestX = x1 + segmentX * t;
      const nearestY = y1 + segmentY * t;
      const dx = pointX - nearestX;
      const dy = pointY - nearestY;

      if (dx * dx + dy * dy <= radiusSquared) {
        setPixel(canvas, x, y, color);
      }
    }
  }
}

function drawBadge(canvas, offsetX = 0) {
  fillCircle(canvas, offsetX + 32, 32, 28, colors.ring);
  fillCircle(canvas, offsetX + 32, 32, 26, colors.badge);
}

function drawSolidM(canvas, offsetX = 0) {
  drawBadge(canvas, offsetX);
  fillRect(canvas, offsetX + 14, 14, 8, 36, colors.mark);
  fillRect(canvas, offsetX + 42, 14, 8, 36, colors.mark);
  fillSegment(canvas, offsetX + 18, 16, offsetX + 32, 34, 9, colors.mark);
  fillSegment(canvas, offsetX + 46, 16, offsetX + 32, 34, 9, colors.mark);
}

function drawSolidMM(canvas, offsetX = 0) {
  drawBadge(canvas, offsetX);
  // Asymmetric fused MM: a dominant left M with a narrower accent M,
  // pushed inward to preserve padding and with deeper inner valleys
  // so the double-letter read survives favicon sizes.
  fillRect(canvas, offsetX + 17, 16, 6, 31, colors.mark);
  fillRect(canvas, offsetX + 29, 18, 5, 29, colors.mark);
  fillRect(canvas, offsetX + 40, 21, 4, 23, colors.mark);
  fillRect(canvas, offsetX + 47, 19, 4, 25, colors.mark);

  fillSegment(canvas, offsetX + 20, 18, offsetX + 25, 33, 6, colors.mark);
  fillSegment(canvas, offsetX + 25, 33, offsetX + 31, 15, 6, colors.mark);
  fillSegment(canvas, offsetX + 31, 15, offsetX + 36, 36, 5, colors.mark);
  fillSegment(canvas, offsetX + 36, 36, offsetX + 41, 23, 5, colors.mark);
  fillSegment(canvas, offsetX + 41, 23, offsetX + 45, 32, 4, colors.mark);
  fillSegment(canvas, offsetX + 45, 32, offsetX + 49, 20, 4, colors.mark);
}

function crc32(buffer) {
  let crc = 0xffffffff;

  for (const byte of buffer) {
    crc ^= byte;

    for (let bit = 0; bit < 8; bit += 1) {
      const mask = -(crc & 1);
      crc = (crc >>> 1) ^ (0xedb88320 & mask);
    }
  }

  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const typeBuffer = Buffer.from(type, "ascii");
  const lengthBuffer = Buffer.alloc(4);
  lengthBuffer.writeUInt32BE(data.length, 0);

  const crcBuffer = Buffer.alloc(4);
  crcBuffer.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);

  return Buffer.concat([lengthBuffer, typeBuffer, data, crcBuffer]);
}

function createPng(rgba, width, height) {
  const signature = Buffer.from([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
  ]);

  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  header[10] = 0;
  header[11] = 0;
  header[12] = 0;

  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);

  for (let y = 0; y < height; y += 1) {
    const rowStart = y * (stride + 1);
    raw[rowStart] = 0;
    rgba.copy(raw, rowStart + 1, y * stride, (y + 1) * stride);
  }

  const compressed = deflateSync(raw);

  return Buffer.concat([
    signature,
    chunk("IHDR", header),
    chunk("IDAT", compressed),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function createIco(pngData, width, height) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(1, 4);

  const directory = Buffer.alloc(16);
  directory[0] = width >= 256 ? 0 : width;
  directory[1] = height >= 256 ? 0 : height;
  directory[2] = 0;
  directory[3] = 0;
  directory.writeUInt16LE(1, 4);
  directory.writeUInt16LE(32, 6);
  directory.writeUInt32LE(pngData.length, 8);
  directory.writeUInt32LE(22, 12);

  return Buffer.concat([header, directory, pngData]);
}

function writeBinary(path, data) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, data);
}

function writePng(path, canvas) {
  writeBinary(path, createPng(Buffer.from(canvas.pixels), canvas.width, canvas.height));
}

const faviconCanvas = createCanvas(iconSize, iconSize);
drawSolidM(faviconCanvas);

const mmPreviewCanvas = createCanvas(iconSize, iconSize);
drawSolidMM(mmPreviewCanvas);

const comparisonCanvas = createCanvas(152, 64, colors.compareBackground);
drawSolidM(comparisonCanvas, 8);
drawSolidMM(comparisonCanvas, 80);
fillRect(comparisonCanvas, 75, 10, 2, 44, colors.compareDivider);

const faviconPng = createPng(
  Buffer.from(faviconCanvas.pixels),
  faviconCanvas.width,
  faviconCanvas.height
);
const faviconIco = createIco(faviconPng, faviconCanvas.width, faviconCanvas.height);

writeBinary(join(root, "src/app/favicon.ico"), faviconIco);
writeBinary(join(root, "public/images/favicon.png"), faviconPng);
writePng(join(root, "docs/favicon-preview.png"), faviconCanvas);
writePng(join(root, "docs/favicon-preview-mm.png"), mmPreviewCanvas);
writePng(join(root, "docs/favicon-preview-compare.png"), comparisonCanvas);
