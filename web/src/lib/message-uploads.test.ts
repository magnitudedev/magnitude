import { afterEach, describe, expect, it, vi } from "vitest"
import { Effect, Either } from "effect"
import { MAX_IMAGE_FILE_UPLOAD_BYTES, MAX_TEXT_FILE_UPLOAD_BYTES } from "@magnitudedev/sdk"
import {
  appendMessageUploads,
  MESSAGE_UPLOAD_ACCEPT,
  ingestClientFile,
  ingestClientFiles,
} from "./message-uploads"

afterEach(() => vi.unstubAllGlobals())

describe("message upload ingestion", () => {
  it("accepts UTF-8 source files as raw text snapshots", async () => {
    const result = await Effect.runPromise(ingestClientFile(
      new File(["export const answer = 42\n"], "answer.ts", { type: "text/plain" }),
    ))

    expect(result.upload).toMatchObject({
      type: "raw_text_file",
      filename: "answer.ts",
    })
    expect(Buffer.from(result.upload.data, "base64").toString("utf8")).toBe("export const answer = 42\n")
  })

  it.each([
    ["empty", new Uint8Array()],
    ["binary", new Uint8Array([0x61, 0x00, 0x62])],
    ["invalid UTF-8", new Uint8Array([0xc3, 0x28])],
  ])("rejects %s files", async (_label, contents) => {
    await expect(Effect.runPromise(ingestClientFile(
      new File([contents], "unsupported.dat"),
    ))).rejects.toBeDefined()
  })

  it("rejects oversized text before reading it", async () => {
    const file = new File([new Uint8Array(MAX_TEXT_FILE_UPLOAD_BYTES + 1)], "large.txt", { type: "text/plain" })
    const result = await Effect.runPromise(Effect.either(ingestClientFile(file)))

    expect(Either.isLeft(result)).toBe(true)
    if (Either.isLeft(result)) {
      expect(result.left.reason).toBe("Text and code files must be 500 KiB or smaller.")
    }
  })

  it("captures supported image dimensions and bytes", async () => {
    const close = vi.fn()
    vi.stubGlobal("createImageBitmap", vi.fn(async () => ({ width: 320, height: 180, close })))
    const result = await Effect.runPromise(ingestClientFile(
      new File([new Uint8Array([1, 2, 3])], "preview.png", { type: "image/png" }),
    ))

    expect(result.upload).toMatchObject({
      type: "raw_image_file",
      filename: "preview.png",
      mediaType: "image/png",
      width: 320,
      height: 180,
    })
    expect(close).toHaveBeenCalledOnce()
  })

  it("accepts SVG as UTF-8 source text rather than a model image", async () => {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="4" /></svg>'
    const result = await Effect.runPromise(ingestClientFile(
      new File([svg], "icon.svg", { type: "image/svg+xml" }),
    ))

    expect(result.upload).toMatchObject({
      type: "raw_text_file",
      filename: "icon.svg",
    })
    expect(Buffer.from(result.upload.data, "base64").toString("utf8")).toBe(svg)
  })

  it("rejects oversized images before reading or decoding them", async () => {
    const file = new File(
      [new Uint8Array(MAX_IMAGE_FILE_UPLOAD_BYTES + 1)],
      "large.png",
      { type: "image/png" },
    )
    const result = await Effect.runPromise(Effect.either(ingestClientFile(file)))

    expect(Either.isLeft(result)).toBe(true)
    if (Either.isLeft(result)) {
      expect(result.left.reason).toBe("Images must be 10 MiB or smaller.")
    }
  })

  it("keeps valid selections when another selected file is rejected", async () => {
    const results = await Effect.runPromise(ingestClientFiles([
      new File(["valid"], "valid.md", { type: "text/markdown" }),
      new File([new Uint8Array([0])], "binary.bin"),
    ]))

    expect(results.map(result => result._tag)).toEqual(["accepted", "rejected"])
  })

  it("advertises the supported image and source families to the native picker", () => {
    expect(MESSAGE_UPLOAD_ACCEPT).toContain("image/png")
    expect(MESSAGE_UPLOAD_ACCEPT).toContain("text/*")
    expect(MESSAGE_UPLOAD_ACCEPT).toContain(".tsx")
    expect(MESSAGE_UPLOAD_ACCEPT).toContain(".md")
    expect(MESSAGE_UPLOAD_ACCEPT).toContain(".svg")
  })
})

describe("message upload capacity", () => {
  const text = (filename: string, byteSize = 1) => ({
    byteSize,
    upload: {
      type: "raw_text_file" as const,
      filename,
      data: Buffer.alloc(byteSize).toString("base64"),
    },
  })

  const image = (filename: string, byteSize: number) => ({
    byteSize,
    upload: {
      type: "raw_image_file" as const,
      filename,
      data: "YQ==",
      mediaType: "image/png" as const,
      width: 1,
      height: 1,
    },
  })

  it("accepts candidates until the per-message count is reached", () => {
    const result = appendMessageUploads([], Array.from({ length: 21 }, (_, index) => text(`${index}.txt`)))
    expect(result.uploads).toHaveLength(20)
    expect(result.rejected).toMatchObject([{
      filename: "20.txt",
      reason: "A message can include at most 20 attachments.",
    }])
  })

  it("rejects the candidate that would exceed the aggregate byte limit", () => {
    const result = appendMessageUploads([], [
      image("first.png", 9 * 1024 * 1024),
      image("second.png", 9 * 1024 * 1024),
      image("third.png", 9 * 1024 * 1024),
    ])
    expect(result.uploads).toHaveLength(2)
    expect(result.rejected).toMatchObject([{
      filename: "third.png",
      reason: "Attachments must total 25 MiB or less per message.",
    }])
  })
})
