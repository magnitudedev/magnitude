import { mkdir, mkdtemp, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, test } from "vitest"
import { Effect } from "effect"
import type { MessageAttachment } from "@magnitudedev/protocol"
import { mergeInlineMentions } from "./file-mentions"

async function makeCwd(): Promise<string> {
  return mkdtemp(join(tmpdir(), "magnitude-file-mentions-"))
}

async function merge(cwd: string, content: string, attachments: MessageAttachment[] = []): Promise<MessageAttachment[]> {
  return Effect.runPromise(mergeInlineMentions(cwd, "", content, attachments))
}

describe("mergeInlineMentions", () => {
  test("adds attachments for inline file-like mentions that resolve under cwd", async () => {
    const cwd = await makeCwd()
    await mkdir(join(cwd, "src"))
    await writeFile(join(cwd, "src", "app.ts"), "const app = true\n")

    const attachments = await merge(cwd, "look at @src/app.ts")

    expect(attachments).toEqual([
      { type: "mention_file", path: "src/app.ts" },
    ])
  })

  test("skips social mentions and unresolved file-like candidates", async () => {
    const cwd = await makeCwd()

    const attachments = await merge(cwd, "thanks @alice and check @missing.ts")

    expect(attachments).toEqual([])
  })

  test("strips trailing prose punctuation only as a fallback", async () => {
    const cwd = await makeCwd()
    await writeFile(join(cwd, "README.md"), "# hi\n")

    const attachments = await merge(cwd, "read @README.md.")

    expect(attachments).toEqual([
      { type: "mention_file", path: "README.md" },
    ])
  })

  test("does not parse mentions inside inline code or fenced code blocks", async () => {
    const cwd = await makeCwd()
    await writeFile(join(cwd, "README.md"), "# hi\n")
    await mkdir(join(cwd, "src"))
    await writeFile(join(cwd, "src", "app.ts"), "const app = true\n")

    const attachments = await merge(cwd, [
      "ignore `@README.md`",
      "```ts",
      "@src/app.ts",
      "```",
    ].join("\n"))

    expect(attachments).toEqual([])
  })

  test("classifies directories and images", async () => {
    const cwd = await makeCwd()
    await mkdir(join(cwd, "docs"))
    await writeFile(join(cwd, "image.png"), "fake")

    const attachments = await merge(cwd, "see @docs/ and @image.png")

    expect(attachments).toEqual([
      { type: "mention_directory", path: "docs/" },
      { type: "mention_file", path: "image.png" },
    ])
  })

  test("dedupes explicit and inline mentions by canonical path and range", async () => {
    const cwd = await makeCwd()
    await mkdir(join(cwd, "src"))
    await writeFile(join(cwd, "src", "app.ts"), "const app = true\n")

    const explicit: MessageAttachment = {
      type: "mention_file",
      path: "src/app.ts",
    }

    const attachments = await merge(cwd, "again @./src/app.ts and ranged @src/app.ts:12", [explicit])

    expect(attachments).toEqual([
      explicit,
      {
        type: "mention_file_range",
        path: "src/app.ts",
        startLine: 2,
        endLine: 22,
      },
    ])
  })

  test("dedupes explicit single-line ranges against inline expanded ranges", async () => {
    const cwd = await makeCwd()
    await mkdir(join(cwd, "src"))
    await writeFile(join(cwd, "src", "app.ts"), "const app = true\n")

    const attachments = await merge(cwd, "same @src/app.ts:12", [{
      type: "mention_file_range",
      path: "./src/app.ts",
      startLine: 12,
      endLine: 12,
    }])

    expect(attachments).toEqual([
      {
        type: "mention_file_range",
        path: "./src/app.ts",
        startLine: 2,
        endLine: 22,
      },
    ])
  })
})
