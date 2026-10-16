import { Either, Option } from "effect"
import { describe, expect, it } from "vitest"
import { drafterInput, parseModelReference, repositoryDisplayName } from "./model-references"

const mac = { platform: "darwin" as const, env: {}, home: "/Users/ada" }
const parse = (input: string) => parseModelReference(input, mac)

describe("model references", () => {
  it("reads owner/repo, a quant, and a file in the repository", () => {
    expect(parse("unsloth/Qwen3.6-GGUF")).toEqual(Either.right({
      _tag: "HuggingFace", repository: "unsloth/Qwen3.6-GGUF", revision: "main", file: Option.none(), quant: Option.none(),
    }))
    expect(parse("unsloth/Qwen3.6-GGUF:Q4_K_M")).toMatchObject(Either.right({ quant: Option.some("Q4_K_M") }))
    expect(parse("z-lab/drafters/sub/dflash.gguf")).toMatchObject(Either.right({ file: Option.some("sub/dflash.gguf") }))
  })

  it("reads Hugging Face links with a pinned revision and file", () => {
    expect(parse("https://huggingface.co/owner/repo/blob/abc123/model-Q4.gguf?download=true")).toEqual(Either.right({
      _tag: "HuggingFace", repository: "owner/repo", revision: "abc123", file: Option.some("model-Q4.gguf"), quant: Option.none(),
    }))
    expect(parse("huggingface.co/owner/repo/tree/dev")).toMatchObject(Either.right({ revision: "dev", file: Option.none() }))
    expect(parse("https://hf.co/owner/repo")).toMatchObject(Either.right({ repository: "owner/repo", revision: "main" }))
  })

  it("reads local paths only when they are explicit", () => {
    expect(parse("~/models/a.gguf")).toEqual(Either.right({ _tag: "Local", path: "/Users/ada/models/a.gguf" }))
    expect(parse("/Volumes/x/a.gguf")).toEqual(Either.right({ _tag: "Local", path: "/Volumes/x/a.gguf" }))
    expect(parseModelReference("C:\\models\\a.gguf", { platform: "win32", env: {}, home: "C:\\Users\\ada" }))
      .toEqual(Either.right({ _tag: "Local", path: "C:\\models\\a.gguf" }))
  })

  it("refuses names that are neither", () => {
    expect(Either.isLeft(parse("not-a-repository"))).toBe(true)
    expect(Either.isLeft(parse("https://example.com/owner/repo"))).toBe(true)
    expect(Either.isLeft(parse("owner/repo:"))).toBe(true)
    expect(Either.isLeft(parse("   "))).toBe(true)
  })

  it("passes an entered drafter to the service as owner/repo[@revision]:path", () => {
    const entered = (input: string) => drafterInput(input, mac)
    expect(entered("magnitudedev/Qwen3.5-4B-DFlash-GGUF/dflash.gguf")).toBe("magnitudedev/Qwen3.5-4B-DFlash-GGUF:dflash.gguf")
    expect(entered("hf:bartowski/Model-GGUF/MTP/mtp-Model-Q8_0.gguf")).toBe("bartowski/Model-GGUF:MTP/mtp-Model-Q8_0.gguf")
    expect(entered("https://huggingface.co/owner/repo/blob/abc123/d.gguf")).toBe("owner/repo@abc123:d.gguf")
    expect(entered("owner/repo:d.gguf")).toBe("owner/repo:d.gguf")
    expect(entered("owner/repo@abc:sub/d.gguf")).toBe("owner/repo@abc:sub/d.gguf")
    expect(entered("owner/repo")).toBe("owner/repo")
    expect(entered("~/drafters/d.gguf")).toBe("/Users/ada/drafters/d.gguf")
  })

  it("derives a readable repository name", () => {
    expect(repositoryDisplayName("HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"))
      .toBe("Qwen3.6 35B A3B Uncensored HauhauCS Aggressive")
    expect(repositoryDisplayName("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")).toBe("Qwen3.6 35B A3B MTP")
  })
})
