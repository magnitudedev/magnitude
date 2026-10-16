import { describe, expect, it } from "vitest"
import { Either, Option } from "effect"
import { deepLinkArgument, isDeepLinkArgument, parseDeepLink } from "./deep-link"

const accepted = (raw: string) => {
  const result = parseDeepLink(raw)
  if (Either.isLeft(result)) throw new Error(`Expected ${raw} to be accepted: ${result.left.message}`)
  return { repository: result.right.repository as string, file: Option.getOrUndefined(result.right.file) as string | undefined }
}

describe("Hugging Face deep links", () => {
  it.each([
    ["magnitude://open_from_hf?model=unsloth/Qwen3.6-27B-GGUF&file=Qwen3.6-27B-Q4_K_M.gguf", "unsloth/Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q4_K_M.gguf"],
    ["magnitude://open_from_hf/?model=bartowski/gemma-4-12b-it-GGUF&file=Q8_0%2Fgemma-4-12b-it-Q8_0-00001-of-00002.gguf", "bartowski/gemma-4-12b-it-GGUF", "Q8_0/gemma-4-12b-it-Q8_0-00001-of-00002.gguf"],
    ["magnitude://open_from_hf?file=model.GGUF&model=owner/repo", "owner/repo", "model.GGUF"],
    ["MAGNITUDE://open_from_hf?model=owner/repo", "owner/repo", undefined],
    ["magnitude://open_from_hf?model=owner%2Frepo&file=a+b.gguf", "owner/repo", "a+b.gguf"],
    ["magnitude://open_from_hf?model=owner/repo&file=x%2By.gguf", "owner/repo", "x+y.gguf"],
  ])("accepts %s", (raw, repository, file) => {
    expect(accepted(raw)).toEqual({ repository, file })
  })

  it.each([
    ["another scheme", "lmstudio://open_from_hf?model=owner/repo"],
    ["an https URL", "https://huggingface.co/owner/repo"],
    ["another action", "magnitude://open_from_disk?model=owner/repo"],
    ["a path", "magnitude://open_from_hf/extra?model=owner/repo"],
    ["credentials", "magnitude://user:pass@open_from_hf?model=owner/repo"],
    ["a port", "magnitude://open_from_hf:80?model=owner/repo"],
    ["a fragment", "magnitude://open_from_hf?model=owner/repo#file=x.gguf"],
    ["no model", "magnitude://open_from_hf?file=model.gguf"],
    ["an empty query", "magnitude://open_from_hf"],
    ["an unknown parameter", "magnitude://open_from_hf?model=owner/repo&revision=main"],
    ["a repeated model", "magnitude://open_from_hf?model=owner/repo&model=other/repo"],
    ["a repeated file", "magnitude://open_from_hf?model=owner/repo&file=a.gguf&file=b.gguf"],
    ["a parameter without a value separator", "magnitude://open_from_hf?model"],
    ["a malformed escape", "magnitude://open_from_hf?model=owner/repo&file=%E0%A4%A.gguf"],
    ["a repository without an owner", "magnitude://open_from_hf?model=repo"],
    ["a nested repository", "magnitude://open_from_hf?model=owner/repo/extra"],
    ["a traversal repository", "magnitude://open_from_hf?model=../repo"],
    ["a dotted repository", "magnitude://open_from_hf?model=owner/..repo"],
    ["a repository with a space", "magnitude://open_from_hf?model=owner/my%20repo"],
    ["a repository with a shell character", "magnitude://open_from_hf?model=owner/repo%3Brm"],
    ["a repository with a quote", "magnitude://open_from_hf?model=owner/repo%22"],
    ["a non-ASCII repository", "magnitude://open_from_hf?model=owner/r%C3%A9po"],
    ["a non-GGUF file", "magnitude://open_from_hf?model=owner/repo&file=model.safetensors"],
    ["an absolute file", "magnitude://open_from_hf?model=owner/repo&file=%2Fetc%2Fmodel.gguf"],
    ["a parent traversal", "magnitude://open_from_hf?model=owner/repo&file=..%2F..%2Fmodel.gguf"],
    ["an encoded parent traversal", "magnitude://open_from_hf?model=owner/repo&file=a%2F%2E%2E%2Fmodel.gguf"],
    ["a current-directory segment", "magnitude://open_from_hf?model=owner/repo&file=.%2Fmodel.gguf"],
    ["an empty segment", "magnitude://open_from_hf?model=owner/repo&file=a%2F%2Fmodel.gguf"],
    ["a backslash", "magnitude://open_from_hf?model=owner/repo&file=a%5Cmodel.gguf"],
    ["a NUL byte", "magnitude://open_from_hf?model=owner/repo&file=model%00.gguf"],
    ["a newline", "magnitude://open_from_hf?model=owner/repo&file=model%0A.gguf"],
    ["a hidden file", "magnitude://open_from_hf?model=owner/repo&file=.model.gguf"],
    ["a file with a space", "magnitude://open_from_hf?model=owner/repo&file=my%20model.gguf"],
    ["a file with a colon", "magnitude://open_from_hf?model=owner/repo&file=C%3Amodel.gguf"],
    ["a raw space", "magnitude://open_from_hf?model=owner/repo --inspect"],
    ["a raw tab", "magnitude://open_from_hf?model=owner/repo\t"],
    ["an empty file", "magnitude://open_from_hf?model=owner/repo&file="],
    ["an overlong link", `magnitude://open_from_hf?model=owner/repo&file=${"a".repeat(2100)}.gguf`],
    ["an overlong segment", `magnitude://open_from_hf?model=owner/repo&file=${"a".repeat(300)}.gguf`],
    ["text that is not a URL", "not a url"],
  ])("rejects %s", (_, raw) => {
    expect(Either.isLeft(parseDeepLink(raw))).toBe(true)
  })

  it("accepts the link the Hugging Face \"Use this model\" entry builds", () => {
    const hub = (model: string, file?: string) => {
      const url = new URL("magnitude://open_from_hf")
      url.searchParams.set("model", model)
      if (file) url.searchParams.set("file", file)
      return url.href
    }
    expect(accepted(hub("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")))
      .toEqual({ repository: "bartowski/Llama-3.2-3B-Instruct-GGUF", file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf" })
    expect(accepted(hub("owner/repo", "Q4_K_M/model+v2-00001-of-00003.gguf"))).toEqual({ repository: "owner/repo", file: "Q4_K_M/model+v2-00001-of-00003.gguf" })
    expect(accepted(hub("owner/repo"))).toEqual({ repository: "owner/repo", file: undefined })
  })

  it("finds the link among Electron launch arguments", () => {
    expect(deepLinkArgument(["C:\\Magnitude\\Magnitude.exe", "magnitude://open_from_hf?model=owner/repo"])).toEqual(Option.some("magnitude://open_from_hf?model=owner/repo"))
    expect(deepLinkArgument(["/usr/lib/magnitude-desktop/magnitude", "--background"])).toEqual(Option.none())
    expect(isDeepLinkArgument("Magnitude://open_from_hf?model=owner/repo")).toBe(true)
    expect(isDeepLinkArgument("/Applications/Magnitude.app")).toBe(false)
  })
})
