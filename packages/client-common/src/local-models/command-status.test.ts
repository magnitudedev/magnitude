import { Cause, FiberId, Option } from "effect"
import { CatalogFormModelIdSchema } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import { localModelCommandStatus, localModelFailureMessage } from "./service"
const first = CatalogFormModelIdSchema.make("first:gguf:q4")
const second = CatalogFormModelIdSchema.make("second:gguf:q4")
describe("model-specific command feedback", () => {
  it("shows the declared failure message without the Effect stack", () => {
    const failure = new Error("Model cleanup could not be verified. Try Stop again.")
    expect(localModelFailureMessage(Cause.fail(failure))).toBe(failure.message)
  })
  it("uses safe wording for defects, interruption, and malformed failures", () => {
    for (const cause of [Cause.die(new Error("private diagnostic")), Cause.interrupt(FiberId.none), Cause.empty, Cause.fail({ message: "" }), Cause.fail("private diagnostic")]) {
      expect(localModelFailureMessage(cause, "Could not read model status. Try again.")).toBe("Could not read model status. Try again.")
    }
  })
  it("does not leak another model's pending command or failure", () => {
    expect(localModelCommandStatus(first, [[{ modelId: second, pending: true, failure: Option.some("Second failed") }]])).toEqual({ pending: false, failures: [] })
  })
  it("replaces a failed invocation when the same model command is retried", () => {
    const prior = { modelId: first, pending: false, failure: Option.some("Download failed") }
    const retry = { modelId: first, pending: true, failure: Option.none<string>() }
    expect(localModelCommandStatus(first, [[prior, retry]])).toEqual({ pending: true, failures: [] })
    expect(localModelCommandStatus(first, [[prior, retry, { ...retry, pending: false }]])).toEqual({ pending: false, failures: [] })
  })
  it("retains independent command outcomes for the same model", () => {
    expect(localModelCommandStatus(first, [
      [{ modelId: first, pending: false, failure: Option.some("Remove failed") }],
      [{ modelId: first, pending: true, failure: Option.none() }],
    ])).toEqual({ pending: true, failures: ["Remove failed"] })
  })
})
