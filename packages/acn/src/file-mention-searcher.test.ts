import { mkdir, mkdtemp, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { describe, expect, test } from "vitest"
import { BunCommandExecutor, BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { DirectoryPathSchema, type RawMentionOccurrence } from "@magnitudedev/acn-protocol"
import { FileMentionSearcher, FileMentionSearcherLive } from "./file-mention-searcher"
import { FileSystemManagerLive } from "./file-system-manager"
import { GitInspectorLive } from "./git-inspector"

const platform = Layer.mergeAll(
  BunFileSystem.layer,
  BunPath.layer,
  BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
)

const searcherLayer = FileMentionSearcherLive.pipe(
  Layer.provide(Layer.mergeAll(
    FileSystemManagerLive.pipe(Layer.provide(platform)),
    GitInspectorLive.pipe(Layer.provide(platform)),
    platform,
  )),
)

async function makeCwd(): Promise<string> {
  return mkdtemp(join(tmpdir(), "magnitude-file-mentions-"))
}

async function collect(
  cwd: string,
  text: string,
  provided: readonly RawMentionOccurrence[] = [],
): Promise<RawMentionOccurrence[]> {
  return Effect.runPromise(
    FileMentionSearcher.pipe(
      Effect.flatMap((searcher) => searcher.collectMentionOccurrences({
        cwd: DirectoryPathSchema.make(cwd),
        scratchpadPath: "",
        content: text,
        provided,
      })),
      Effect.map((occurrences) => [...occurrences]),
      Effect.provide(searcherLayer),
    ),
  )
}

describe("collectMentionOccurrences", () => {
  test("discovers every inline occurrence and preserves its UTF-16 span", async () => {
    const cwd = await makeCwd()
    await mkdir(join(cwd, "src"))
    await writeFile(join(cwd, "src", "app.ts"), "const app = true\n")
    const text = "😀 see @src/app.ts, then @src/app.ts"

    const mentions = await collect(cwd, text)

    expect(mentions).toHaveLength(2)
    expect(mentions.map(({ attachment, placement }) => ({ attachment, placement }))).toEqual([
      {
        attachment: { type: "mention_file", path: "src/app.ts" },
        placement: { _tag: "inline", start: text.indexOf("@src/app.ts"), end: text.indexOf("@src/app.ts") + 11 },
      },
      {
        attachment: { type: "mention_file", path: "src/app.ts" },
        placement: { _tag: "inline", start: text.lastIndexOf("@src/app.ts"), end: text.lastIndexOf("@src/app.ts") + 11 },
      },
    ])
    expect(mentions[0]?.occurrenceId).not.toBe(mentions[1]?.occurrenceId)
  })

  test("does not discover mentions inside inline or fenced code", async () => {
    const cwd = await makeCwd()
    await writeFile(join(cwd, "README.md"), "# hi\n")
    const text = ["ignore `@README.md`", "```", "@README.md", "```"].join("\n")
    expect(await collect(cwd, text)).toEqual([])
  })

  test("keeps explicit trailing mentions after ordered inline mentions", async () => {
    const cwd = await makeCwd()
    await writeFile(join(cwd, "README.md"), "# hi\n")
    const text = "read @README.md"
    const trailing: RawMentionOccurrence = {
      occurrenceId: "trailing-1",
      attachment: { type: "mention_file", path: "README.md" },
      placement: { _tag: "trailing" },
    }

    const mentions = await collect(cwd, text, [trailing])

    expect(mentions).toHaveLength(2)
    expect(mentions[0]?.placement._tag).toBe("inline")
    expect(mentions[1]).toEqual(trailing)
  })

  test("rejects overlapping or invalid explicit spans as a typed placement error", async () => {
    const cwd = await makeCwd()
    await writeFile(join(cwd, "README.md"), "# hi\n")
    const occurrence: RawMentionOccurrence = {
      occurrenceId: "bad-span",
      attachment: { type: "mention_file", path: "README.md" },
      placement: { _tag: "inline", start: 0, end: 4 },
    }

    const result = await Effect.runPromise(
      FileMentionSearcher.pipe(
        Effect.flatMap((searcher) => searcher.collectMentionOccurrences({
          cwd: DirectoryPathSchema.make(cwd),
          scratchpadPath: "",
          content: "read @README.md",
          provided: [occurrence],
        })),
        Effect.either,
        Effect.provide(searcherLayer),
      ),
    )
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("InvalidMentionPlacement")
  })
})
