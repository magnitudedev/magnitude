import { describe, expect, it } from "vitest"
import { normalizeLlamaModelPath } from "./identity"

describe("normalizeLlamaModelPath", () => {
  it.each([
    ["/models/./qwen.gguf", "/models/qwen.gguf"],
    ["/models/a/../qwen.gguf/", "/models/qwen.gguf"],
    ["c:\\models\\qwen.gguf", "C:/models/qwen.gguf"],
    ["c:\\", "C:/"],
    ["//server//models/qwen.gguf", "//server/models/qwen.gguf"],
    ["relative/../qwen.gguf", "qwen.gguf"],
  ])("normalizes %s", (input, expected) => {
    expect(normalizeLlamaModelPath(input)).toBe(expected)
  })

  it.each(["", "   ", "none", "a\0b"])("rejects %j", (input) => {
    expect(normalizeLlamaModelPath(input)).toBeUndefined()
  })
})
