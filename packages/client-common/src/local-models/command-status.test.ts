import { Cause, FiberId, Option } from "effect"
import { CatalogFormModelIdSchema, LocalModelMutationFailed } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import { localModelCommandStatus, localModelCommandFailure, type LocalModelCommand } from "./service"
import { makeSetupModel } from "../application/fixtures/model"
const first = CatalogFormModelIdSchema.make("first:gguf:q4")
const second = CatalogFormModelIdSchema.make("second:gguf:q4")
const failure = (operation: LocalModelCommand, code = "low_memory") => localModelCommandFailure(operation, Cause.fail(new LocalModelMutationFailed({ code, message: "private diagnostics", retryable: true })))
const observation = (operation: LocalModelCommand, pending = false, modelId = first) => ({ modelId, operation, pending, failure: pending ? Option.none() : Option.some(failure(operation)) })
const empty = { pending: false, pendingOperations: [], failures: [] }
describe("model-specific command feedback", () => {
  it("preserves structured rejection facts and does not treat unknown errors as classified failures", () => {
    expect(Option.getOrThrow(failure("load").rejection).code).toBe("low_memory")
    for (const cause of [Cause.die("private diagnostics"), Cause.interrupt(FiberId.none), Cause.fail(new Error("private diagnostics"))]) {
      expect(localModelCommandFailure("load", cause)).toEqual({ operation: "load", rejection: Option.none() })
    }
  })
  it("does not leak another model's command", () => {
    expect(localModelCommandStatus(first, [[observation("load", true, second)]], Option.none())).toEqual(empty)
  })
  it("replaces only the retried command's failure", () => {
    const prior = observation("load")
    const retry = observation("load", true)
    expect(localModelCommandStatus(first, [[prior, retry]], Option.none())).toEqual({ ...empty, pending: true, pendingOperations: ["load"] })
    expect(localModelCommandStatus(first, [[prior, retry, { ...retry, pending: false }]], Option.none())).toEqual(empty)
    expect(localModelCommandStatus(first, [[observation("remove")], [retry]], Option.none())).toEqual({ ...empty, pending: true, pendingOperations: ["load"], failures: [failure("remove")] })
  })
  it("retires load feedback when the model is observed ready without erasing other commands", () => {
    const base = makeSetupModel(true)
    const model = { ...base, modelId: first, acquisitionState: { ...base.acquisitionState, residencyState: { _tag: "Ready" as const, allocation: { contextWindowTokens: 4096, memoryDomains: [] } } } }
    expect(localModelCommandStatus(first, [[observation("load")], [observation("remove")]], Option.some(model)).failures).toEqual([failure("remove")])
  })
  it("shows removal failure once while retaining a separate load rejection", () => {
    const base = makeSetupModel(true)
    if (base.acquisitionState._tag !== "Installed") throw new Error("Expected installed fixture")
    const rejection = failure("remove", "model_removal_retained_shared")
    const model = { ...base, modelId: first, acquisitionState: { ...base.acquisitionState, _tag: "RemoveFailed" as const, failure: Option.getOrThrow(rejection.rejection) } }
    const remove = { ...observation("remove"), failure: Option.some(rejection) }
    expect(localModelCommandStatus(first, [[remove], [observation("load")]], Option.some(model)).failures).toEqual([failure("load")])
  })
  it("prefers an authoritative load failure only for the matching rejection category", () => {
    const base = makeSetupModel(true)
    const model = { ...base, modelId: first, acquisitionState: { ...base.acquisitionState, residencyState: { _tag: "Failed" as const, failure: { code: "low_memory", message: "native message", retryable: true } } } }
    expect(localModelCommandStatus(first, [[observation("load")]], Option.some(model))).toEqual(empty)
    const transport = { ...observation("load"), failure: Option.some(failure("load", "model_load_transport_failed")) }
    expect(localModelCommandStatus(first, [[transport]], Option.some(model)).failures).toEqual([failure("load", "model_load_transport_failed")])
    expect(localModelCommandStatus(first, [[observation("load")]], Option.none()).failures).toEqual([failure("load")])
    expect(localModelCommandStatus(first, [[observation("remove")]], Option.some(model)).failures).toEqual([failure("remove")])
  })
})
