import { Effect, Either, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  absolutePath,
  decodeCustomModelId,
  drafterReferenceArgument,
  isLocalPathReference,
  parseDrafterArgument,
  parseHuggingFaceReference,
  parsePullReference,
} from "./model-references"

const pullReference = (input: string) => Effect.runSync(Effect.either(parsePullReference(input)))

describe("model references", () => {
  it("parses catalog IDs and Hugging Face repositories for pull", () => {
    expect(pullReference("qwen-test:gguf:q4")).toEqual(Either.right({ _tag: "Catalog", modelId: "qwen-test:gguf:q4" }))
    expect(pullReference("HauhauCS/Qwen-GGUF:Q4_K_P")).toEqual(Either.right({
      _tag: "HuggingFace",
      reference: { repository: "HauhauCS/Qwen-GGUF", revision: Option.none(), version: Option.some("Q4_K_P"), file: Option.none() },
    }))
    expect(pullReference("HauhauCS/Qwen-GGUF")).toMatchObject(Either.right({ reference: { version: Option.none() } }))
    expect(Either.isLeft(pullReference("owner/repo:"))).toBe(true)
    expect(Either.isLeft(pullReference("owner"))).toBe(true)
    expect(Either.isLeft(pullReference("a/b/c"))).toBe(true)
  })

  it("parses huggingface.co links with revisions and files", () => {
    expect(parseHuggingFaceReference("https://huggingface.co/unsloth/Qwen-GGUF")).toEqual(Option.some({
      repository: "unsloth/Qwen-GGUF", revision: Option.none(), version: Option.none(), file: Option.none(),
    }))
    expect(parseHuggingFaceReference("https://huggingface.co/unsloth/Qwen-GGUF/tree/abc123")).toMatchObject(Option.some({
      revision: Option.some("abc123"), file: Option.none(),
    }))
    expect(parseHuggingFaceReference("https://hf.co/unsloth/Qwen-GGUF/resolve/main/Q4/model-00001-of-00002.gguf")).toMatchObject(Option.some({
      revision: Option.some("main"), file: Option.some("Q4/model-00001-of-00002.gguf"),
    }))
    expect(parseHuggingFaceReference("https://example.com/unsloth/Qwen-GGUF")).toEqual(Option.none())
  })

  it("tells local paths from Hugging Face references", () => {
    for (const path of ["/models/q.gguf", "./q.gguf", "../q.gguf", "~/models/q.gguf", "q.gguf", "models/q.gguf", "C:\\models\\q.gguf"]) {
      expect(isLocalPathReference(path), path).toBe(true)
    }
    for (const reference of ["owner/repo", "owner/repo:Q4", "owner/repo/drafter.gguf", "https://huggingface.co/a/b/blob/main/x.gguf"]) {
      expect(isLocalPathReference(reference), reference).toBe(false)
    }
    expect(absolutePath("models/q.gguf", "/work")).toBe("/work/models/q.gguf")
    expect(absolutePath("/abs/q.gguf", "/work")).toBe("/abs/q.gguf")
    expect(absolutePath("~/q.gguf", "/work")).not.toContain("~")
  })

  it("parses drafter arguments and prints them back", () => {
    const parse = (input: string) => Effect.runSync(parseDrafterArgument(input, "/work"))
    expect(parse("AUTO")).toEqual({ _tag: "Auto" })
    expect(parse("none")).toEqual({ _tag: "None" })
    expect(parse("built-in")).toEqual({ _tag: "Reference", reference: { _tag: "Embedded" } })
    expect(parse("qwen-test:gguf:q4")).toEqual({ _tag: "Reference", reference: { _tag: "Catalog", modelId: "qwen-test:gguf:q4" } })
    expect(parse("z-lab/Qwen-DFlash")).toEqual({ _tag: "Lookup", reference: "z-lab/Qwen-DFlash" })
    expect(parse("drafters/mtp.gguf")).toEqual({ _tag: "Lookup", reference: "/work/drafters/mtp.gguf" })
    expect(drafterReferenceArgument({ _tag: "Embedded" })).toBe("built-in")
    expect(drafterReferenceArgument({ _tag: "Local", path: "/d.gguf" })).toBe("/d.gguf")
  })

  it("refuses catalog IDs where only custom models apply", () => {
    const result = Effect.runSync(Effect.either(decodeCustomModelId("qwen-test:gguf:q4", (id) => `no ${id}`)))
    expect(result).toMatchObject(Either.left({ message: "no qwen-test:gguf:q4" }))
    expect(Effect.runSync(Effect.either(decodeCustomModelId(`local:${"e".repeat(64)}`, () => "")))).toEqual(Either.right(`local:${"e".repeat(64)}`))
  })
})
