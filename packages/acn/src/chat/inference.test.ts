import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { inferenceFailed } from "./inference"

type ClientError = Parameters<typeof inferenceFailed>[0]

describe("inference errors", () => {
  it("keeps the engine's error as it reported it", () => {
    const failure = inferenceFailed({
      _tag: "GeneratedClientRemoteError",
      body: {
        error: {
          type: "invalid_request_error",
          code: "context_length_exceeded",
          message: "Prompt exceeds the configured context window",
          param: Option.some(null),
        },
      },
    } as ClientError)
    expect(failure.error).toEqual({
      _tag: "inference",
      code: "context_length_exceeded",
      message: "Prompt exceeds the configured context window",
      param: Option.none(),
    })
  })

  it("reports a service it could not reach, or whose stream broke, as unreachable", () => {
    expect(inferenceFailed({ _tag: "GeneratedClientTransportError", cause: new Error("Connection closed") } as ClientError).error)
      .toEqual({ _tag: "unreachable", message: "The inference service could not be reached: Connection closed" })
    expect(inferenceFailed({ _tag: "GeneratedClientIncompleteStreamError" } as ClientError).error._tag).toBe("unreachable")
    expect(inferenceFailed({ _tag: "GeneratedClientInvalidResponseError", message: "Bad event" } as ClientError).error)
      .toEqual({ _tag: "unreachable", message: "The inference service sent a malformed response: Bad event" })
  })

  it("reports a request the client refused to send as rejected", () => {
    expect(inferenceFailed({ _tag: "GeneratedClientInputError" } as ClientError).error._tag).toBe("rejected")
  })
})
