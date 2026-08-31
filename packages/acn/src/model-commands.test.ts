import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { modelCommandFailure } from "./model-commands"

describe("modelCommandFailure", () => {
  it("preserves the server's structured remote failure details", () => {
    const failure = modelCommandFailure("load_model", {
      _tag: "GeneratedClientRemoteError",
      operationId: "ensureModelInstance",
      status: 409,
      headers: {},
      body: {
        error: {
          code: "low_memory",
          message: "Not enough memory to load the selected model",
          type: "model_error",
          param: Option.none(),
        },
      },
    } as never)

    expect(failure).toMatchObject({
      _tag: "LocalModelMutationFailed",
      code: "low_memory",
      message: "Not enough memory to load the selected model",
      retryable: true,
    })
  })

  it("uses the underlying transport message instead of the wrapper tag", () => {
    const failure = modelCommandFailure("load_model", {
      _tag: "GeneratedClientTransportError",
      operationId: "ensureModelInstance",
      cause: new Error("ICN connection closed while loading"),
    } as never)

    expect(failure).toMatchObject({
      _tag: "LocalModelMutationFailed",
      code: "model_load_model_transport_failed",
      message: "ICN connection closed while loading",
      retryable: true,
    })
  })
})
