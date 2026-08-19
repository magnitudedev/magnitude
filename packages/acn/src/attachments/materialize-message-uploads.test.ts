import { afterEach, describe, expect, it } from "vitest"
import * as BunFileSystem from "@effect/platform-bun/BunFileSystem"
import * as BunPath from "@effect/platform-bun/BunPath"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Layer } from "effect"
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { materializeMessageUploads } from "./materialize-message-uploads"

const roots: string[] = []
const platformLayer = Layer.mergeAll(BunFileSystem.layer, BunPath.layer)

afterEach(async () => Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))))

async function scratchpad(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "magnitude-message-uploads-"))
  roots.push(root)
  return root
}

const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | Path.Path>) =>
  Effect.runPromise(effect.pipe(Effect.provide(platformLayer)))

describe("materializeMessageUploads", () => {
  it("materializes mixed image and text uploads into their authoritative forms", async () => {
    const root = await scratchpad()
    const png = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    )
    const result = await run(materializeMessageUploads({
      scratchpadPath: root,
      uploads: [
        {
          type: "raw_image_file",
          filename: "pixel.png",
          data: png.toString("base64"),
          mediaType: "image/png",
          width: 1,
          height: 1,
        },
        {
          type: "raw_text_file",
          filename: "notes.md",
          data: Buffer.from("# Notes\n").toString("base64"),
        },
      ],
    }))

    expect(result.images).toHaveLength(1)
    expect(result.images[0]?.path).toBe("$M/attachments/pixel.png")
    expect(result.images[0]?.mediaType).toBe("image/png")
    expect(result.images[0]?.dimensions).toEqual({ width: 1, height: 1 })
    expect(result.trailingMentions).toEqual([expect.objectContaining({
      attachment: { type: "mention_file", path: "$M/attachments/notes.md" },
      placement: { _tag: "trailing" },
    })])
    expect(await readFile(join(root, "attachments", "pixel.png"))).toEqual(png)
    expect(await readFile(join(root, "attachments", "notes.md"), "utf8")).toBe("# Notes\n")
  })

  it("captures UTF-8 text as a trailing scratchpad mention", async () => {
    const root = await scratchpad()
    const result = await run(materializeMessageUploads({
      scratchpadPath: root,
      uploads: [{
        type: "raw_text_file",
        filename: "notes.md",
        data: Buffer.from("# Notes\n\nHello.\n").toString("base64"),
      }],
    }))

    expect(result.images).toEqual([])
    expect(result.trailingMentions).toHaveLength(1)
    expect(result.trailingMentions[0]).toMatchObject({
      attachment: { type: "mention_file", path: "$M/attachments/notes.md" },
      placement: { _tag: "trailing" },
    })
    expect(await readFile(join(root, "attachments", "notes.md"), "utf8")).toBe("# Notes\n\nHello.\n")
  })

  it("preserves every upload when selected files have duplicate names", async () => {
    const root = await scratchpad()
    const result = await run(materializeMessageUploads({
      scratchpadPath: root,
      uploads: ["first", "second"].map(contents => ({
        type: "raw_text_file" as const,
        filename: "notes.md",
        data: Buffer.from(contents).toString("base64"),
      })),
    }))

    expect(result.trailingMentions.map(mention => mention.attachment.path)).toEqual([
      "$M/attachments/notes.md",
      "$M/attachments/notes-1.md",
    ])
    expect(await readFile(join(root, "attachments", "notes.md"), "utf8")).toBe("first")
    expect(await readFile(join(root, "attachments", "notes-1.md"), "utf8")).toBe("second")
  })

  it("atomically preserves same-name uploads materialized concurrently", async () => {
    const root = await scratchpad()
    const results = await run(Effect.all(
      ["first", "second"].map(contents => materializeMessageUploads({
        scratchpadPath: root,
        uploads: [{
          type: "raw_text_file" as const,
          filename: "notes.md",
          data: Buffer.from(contents).toString("base64"),
        }],
      })),
      { concurrency: "unbounded" },
    ))

    expect(results.map(result => result.trailingMentions[0]?.attachment.path).sort()).toEqual([
      "$M/attachments/notes-1.md",
      "$M/attachments/notes.md",
    ])
    expect((await Promise.all([
      readFile(join(root, "attachments", "notes.md"), "utf8"),
      readFile(join(root, "attachments", "notes-1.md"), "utf8"),
    ])).sort()).toEqual(["first", "second"])
  })

  it("rejects attachment counts and aggregate bytes above the message limits", async () => {
    const countRoot = await scratchpad()
    await expect(run(materializeMessageUploads({
      scratchpadPath: countRoot,
      uploads: Array.from({ length: 21 }, (_, index) => ({
        type: "raw_text_file" as const,
        filename: `${index}.txt`,
        data: "YQ==",
      })),
    }))).rejects.toThrow("A message can include at most 20 attachments")

    const sizeRoot = await scratchpad()
    const nineMiB = Buffer.alloc(9 * 1024 * 1024, 0x61).toString("base64")
    await expect(run(materializeMessageUploads({
      scratchpadPath: sizeRoot,
      uploads: Array.from({ length: 3 }, (_, index) => ({
        type: "raw_image_file" as const,
        filename: `${index}.png`,
        data: nineMiB,
        mediaType: "image/png" as const,
        width: 1,
        height: 1,
      })),
    }))).rejects.toThrow("Attachments exceed the 25 MiB total message limit")
  })

  it.each([
    ["empty", Buffer.alloc(0).toString("base64")],
    ["NUL-containing", Buffer.from([0x61, 0x00, 0x62]).toString("base64")],
    ["invalid UTF-8", Buffer.from([0xc3, 0x28]).toString("base64")],
    ["non-canonical base64", "not base64"],
  ])("rejects %s text without writing a file", async (_label, data) => {
    const root = await scratchpad()
    await expect(run(materializeMessageUploads({
      scratchpadPath: root,
      uploads: [{ type: "raw_text_file", filename: "bad.txt", data }],
    }))).rejects.toBeDefined()
    await expect(readdir(join(root, "attachments"))).rejects.toMatchObject({ code: "ENOENT" })
  })

  it("removes files materialized earlier in the batch when a later image is invalid", async () => {
    const root = await scratchpad()
    await expect(run(materializeMessageUploads({
      scratchpadPath: root,
      uploads: [
        {
          type: "raw_text_file",
          filename: "valid.txt",
          data: Buffer.from("valid").toString("base64"),
        },
        {
          type: "raw_image_file",
          filename: "broken.png",
          data: Buffer.from("not an image").toString("base64"),
          mediaType: "image/png",
          width: 1,
          height: 1,
        },
      ],
    }))).rejects.toBeDefined()

    expect(await readdir(join(root, "attachments"))).toEqual([])
  })

  it("rejects an oversized image without writing it", async () => {
    const root = await scratchpad()
    await expect(run(materializeMessageUploads({
      scratchpadPath: root,
      uploads: [{
        type: "raw_image_file",
        filename: "large.png",
        data: Buffer.alloc((10 * 1024 * 1024) + 1).toString("base64"),
        mediaType: "image/png",
        width: 1,
        height: 1,
      }],
    }))).rejects.toBeDefined()

    await expect(readdir(join(root, "attachments"))).rejects.toMatchObject({ code: "ENOENT" })
  })
})
