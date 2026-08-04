import { Effect } from "effect"
import { describe, expect, it } from "vitest"

import {
  formatLocalModelEvaluationFailure,
  localModelAssessmentFailure,
  localModelAssessmentResultFromIcn,
} from "./local-model-evaluations"

describe("localModelAssessmentResultFromIcn", () => {
  it("preserves a typed assessment failure without native diagnostics", () => {
    const result = Effect.runSync(localModelAssessmentResultFromIcn({
      _tag: "AssessmentFailed",
      requestId: "assessment-0",
      targetId: "target-0",
      failure: {
        code: "planner_timeout",
        message: "Hardware assessment took longer than five minutes.",
        retryable: true,
      },
    }))

    expect(result).toEqual({
      _tag: "AssessmentFailed",
      failure: {
        code: "planner_timeout",
        message: "Hardware assessment took longer than five minutes.",
        retryable: true,
      },
    })
    expect(JSON.stringify(result)).not.toContain("isolated native planner")
  })
})

describe("localModelAssessmentFailure", () => {
  it("returns the first operational failure in a batch", () => {
    expect(localModelAssessmentFailure([{
      _tag: "InvalidTarget",
      message: "Invalid package",
    }, {
      _tag: "AssessmentFailed",
      failure: {
        code: "planner_timeout",
        message: "Hardware assessment took longer than five minutes.",
        retryable: true,
      },
    }])).toEqual({
      code: "planner_timeout",
      message: "Hardware assessment took longer than five minutes.",
      retryable: true,
    })
  })

  it("does not classify invalid targets as operational failures", () => {
    expect(localModelAssessmentFailure([{
      _tag: "InvalidTarget",
      message: "Invalid package",
    }])).toBeUndefined()
  })
})

describe("formatLocalModelEvaluationFailure", () => {
  it("preserves structured remote error details for internal diagnostics", () => {
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
