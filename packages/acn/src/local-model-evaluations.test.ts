import { describe, expect, it } from "vitest"

import { formatLocalModelEvaluationFailure } from "./local-model-evaluations"

describe("formatLocalModelEvaluationFailure", () => {
  it("preserves structured remote error details", () => {
    const detail = formatLocalModelEvaluationFailure({
      _tag: "GeneratedClientRemoteError",
      operationId: "assessModels",
      status: 500,
      body: {
        error: {
          code: "inventory_error",
          message: "isolated native planner exited with status 1",
          retryable: true,
          type: "server_error",
        },
      },
    })

    expect(detail).toContain("GeneratedClientRemoteError")
    expect(detail).toContain("assessModels")
    expect(detail).toContain("inventory_error")
    expect(detail).toContain("isolated native planner exited with status 1")
  })

  it("preserves ordinary error stacks", () => {
    const detail = formatLocalModelEvaluationFailure(new Error("hardware assessment failed"))

    expect(detail).toContain("Error: hardware assessment failed")
    expect(detail).toContain("local-model-evaluations.test.ts")
  })

  it("preserves nested error causes", () => {
    const detail = formatLocalModelEvaluationFailure({
      _tag: "GeneratedClientTransportError",
      operationId: "assessModels",
      cause: new Error("connection failed"),
    })

    expect(detail).toContain("GeneratedClientTransportError")
    expect(detail).toContain("Error: connection failed")
    expect(detail).toContain("local-model-evaluations.test.ts")
  })
})
