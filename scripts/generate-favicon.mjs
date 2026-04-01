import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

const size = 64;
const pixels = new Uint8Array(size * size * 4);

const colors = {
  transparent: [0, 0, 0, 0],
  badge: [11, 15, 18, 255],
  ring: [24, 33, 43, 255],
  cyan: [130, 212, 255, 255],
  purple: [170, 140, 250, 255],
  highlight: [232, 239, 249, 255],
};

function setPixel(x, y, color) {
  if (x < 0 || x >= size || y < 0 || y >= size) {
    return;
  }

  const index = (y * size + x) * 4;
  pixels[index] = color[0];
  pixels[index + 1] = color[1];
  pixels[index + 2] = color[2];
  pixels[index + 3] = color[3];
}

function fillCircle(centerX, centerY, radius, color) {
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
        setPixel(x, y, color);
      }
    }
  }
}

function fillRect(startX, startY, width, height, color) {
  const endX = startX + width;
  const endY = startY + height;

  for (let y = startY; y < endY; y += 1) {
    for (let x = startX; x < endX; x += 1) {
      setPixel(x, y, color);
    }
  }
}

function fillSegment(x1, y1, x2, y2, thickness, color) {
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
        setPixel(x, y, color);
      }
    }
  }
}

fillCircle(size / 2, size / 2, 28, colors.ring);
fillCircle(size / 2, size / 2, 26, colors.badge);
fillRect(14, 16, 4, 32, colors.cyan);
fillRect(46, 16, 4, 32, colors.purple);
fillSegment(16, 18, 28, 40, 7, colors.cyan);
fillSegment(48, 18, 36, 40, 7, colors.purple);
fillSegment(28, 40, 32, 31, 6, colors.highlight);
fillSegment(32, 31, 36, 40, 6, colors.highlight);
fillCircle(32, 14, 3, colors.highlight);

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

const rgbaBuffer = Buffer.from(pixels);
const png = createPng(rgbaBuffer, size, size);
const ico = createIco(png, size, size);

writeBinary(join(root, "docs/favicon-preview.png"), png);
writeBinary(join(root, "public/images/favicon.png"), png);
writeBinary(join(root, "src/app/favicon.ico"), ico);
