import { writeFile } from "node:fs/promises";
import {
  decode as wasmDecode,
  dimensions as wasmDimensions,
  format as wasmFormat,
  render_svg as wasmRenderSvg,
  resize as wasmResize,
  encode_png as wasmEncodePng,
  encode_jpeg as wasmEncodeJpeg,
  pixel_diff as wasmPixelDiff,
} from "../pkg/magnitude_image.js";

export interface ImageData {
  data: Uint8Array;
  width: number;
  height: number;
}

type EncodedInput = Buffer | Uint8Array;

function toUint8Array(buf: EncodedInput): Uint8Array {
  return buf instanceof Uint8Array ? buf : new Uint8Array(buf);
}

export class Image {
  private rgba: Uint8Array;
  private _width: number;
  private _height: number;

  private constructor(rgba: Uint8Array, width: number, height: number) {
    this.rgba = rgba;
    this._width = width;
    this._height = height;
  }

  static fromBuffer(buf: EncodedInput): Image {
    const decoded = wasmDecode(toUint8Array(buf)) as ImageData;
    return new Image(new Uint8Array(decoded.data), decoded.width, decoded.height);
  }

  static fromBase64(base64: string): Image {
    return Image.fromBuffer(Buffer.from(base64, "base64"));
  }

  static fromSvg(
    svgData: Buffer | Uint8Array | string,
    options?: { maxWidth?: number; maxHeight?: number },
  ): Image {
    const bytes =
      typeof svgData === "string" ? new TextEncoder().encode(svgData) : toUint8Array(svgData);
    const maxWidth = options?.maxWidth ?? 1568;
    const maxHeight = options?.maxHeight ?? 1568;
    const decoded = wasmRenderSvg(bytes, maxWidth, maxHeight) as ImageData;
    return new Image(new Uint8Array(decoded.data), decoded.width, decoded.height);
  }

  get width(): number {
    return this._width;
  }

  get height(): number {
    return this._height;
  }

  resize(width: number, height: number): Image {
    const resized = wasmResize(this.rgba, this._width, this._height, width, height);
    return new Image(new Uint8Array(resized), width, height);
  }

  toPng(): Buffer {
    return Buffer.from(wasmEncodePng(this.rgba, this._width, this._height));
  }

  toJpeg(quality = 80): Buffer {
    return Buffer.from(wasmEncodeJpeg(this.rgba, this._width, this._height, quality));
  }

  toBase64(format: "png" | "jpeg" = "png"): string {
    return this.toBuffer(format).toString("base64");
  }

  toBuffer(format: "png" | "jpeg" = "png"): Buffer {
    return format === "jpeg" ? this.toJpeg() : this.toPng();
  }

  async saveToFile(path: string): Promise<void> {
    await writeFile(path, this.toPng());
  }

  rawPixels(): Uint8Array {
    return new Uint8Array(this.rgba);
  }
}

export function dimensions(buf: EncodedInput): { width: number; height: number } {
  return wasmDimensions(toUint8Array(buf)) as { width: number; height: number };
}

export function format(buf: EncodedInput): string {
  return wasmFormat(toUint8Array(buf));
}

export function pixelDiff(img1: Image, img2: Image): number {
  if (img1.width !== img2.width || img1.height !== img2.height) {
    throw new Error("Images must have the same dimensions for pixel diff");
  }
  return wasmPixelDiff(img1.rawPixels(), img2.rawPixels(), img1.width, img1.height);
}