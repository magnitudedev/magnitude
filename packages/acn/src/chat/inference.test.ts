import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { chatInferenceError } from "./inference"
describe("Chat inference errors", () => {
  it("preserves the engine's actionable error from an already-decoded response", () => {
    expect(
      chatInferenceError({
        _tag: "GeneratedClientRemoteError",
        body: {
          error: {
            message: "Prompt exceeds the configured context window",
            code: Option.some("context_length_exceeded"),
            details: Option.none(),
          },
        },
      }).message
    ).toBe("Prompt exceeds the configured context window")
  })
  it("bounds malformed error handling to a useful fallback", () => {
    expect(
      chatInferenceError({ _tag: "GeneratedClientRemoteError", body: {} })
        .message
    ).toContain("Check Status")
    expect(chatInferenceError(new Error("Connection closed")).message).toBe(
      "Connection closed"
    )
  })
})
