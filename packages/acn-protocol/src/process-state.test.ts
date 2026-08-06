import * as FileSystem from "@effect/platform/FileSystem"
import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Either, Option, Scope } from "effect"
import { describe, expect, it } from "vitest"
import { AcnIdentitySchema, ProcessStartIdentitySchema } from "./acn-identity"
import {
  AcnProcessStateConflict,
  applyAcnProcessCommand,
  readAcnProcessState,
  reduceAcnProcessState,
  type AcnProcessState,
  type ExactProcess,
} from "./process-state"

const identity = (value: string) => AcnIdentitySchema.make(value)
const manager: ExactProcess = {
  pid: 100,
  processStartIdentity: ProcessStartIdentitySchema.make("manager-100"),
}
const candidate = {
  identity: identity("1.0.0"),
  pid: 200,
  processStartIdentity: ProcessStartIdentitySchema.make("candidate-200"),
}

const spawnedCandidate = () => {
  const begun = reduceAcnProcessState(Option.none(), {
    _tag: "BeginEnsure" as const,
    target: identity("1.0.0"),
    manager,
  })
  const prepared = reduceAcnProcessState(Option.some(begun), {
    _tag: "PreparationSucceeded",
    manager,
  })
  return reduceAcnProcessState(Option.some(prepared), {
    _tag: "CandidateSpawned",
    manager,
    candidate,
  })
}

const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | Scope.Scope>) =>
  Effect.runPromise(effect.pipe(Effect.scoped, Effect.provide(BunFileSystem.layer)))

describe("ACN process state", () => {
  it("retains the incumbent until replacement preparation succeeds", () => {
    const spawned = spawnedCandidate()
    const assigned = reduceAcnProcessState(Option.some(spawned), {
      _tag: "CandidateAdmitted",
      candidate,
      id: "candidate" as never,
      url: "http://127.0.0.1:1234",
    })
    if (assigned.mode._tag !== "Assigned") throw new Error("unreachable")
    const preparing = reduceAcnProcessState(Option.some(assigned), {
      _tag: "BeginReplacement",
      target: identity("2.0.0"),
      manager,
      current: assigned.mode.current,
    })
    expect(preparing.mode).toMatchObject({
      _tag: "Changing",
      owner: { _tag: "Manager", phase: { _tag: "Preparing", current: Option.some(assigned.mode.current) } },
    })
    const prepared = reduceAcnProcessState(Option.some(preparing), {
      _tag: "PreparationSucceeded",
      manager,
    })
    expect(prepared.mode).toMatchObject({
      _tag: "Changing",
      owner: { _tag: "Manager", phase: { _tag: "RetiringAssigned", current: assigned.mode.current } },
    })
  })

  it("restores the exact incumbent when replacement preparation fails", () => {
    const spawned = spawnedCandidate()
    const assigned = reduceAcnProcessState(Option.some(spawned), {
      _tag: "CandidateAdmitted",
      candidate,
      id: "candidate" as never,
      url: "http://127.0.0.1:1234",
    })
    if (assigned.mode._tag !== "Assigned") throw new Error("unreachable")
    const preparing = reduceAcnProcessState(Option.some(assigned), {
      _tag: "BeginReplacement",
      target: identity("2.0.0"),
      manager,
      current: assigned.mode.current,
    })
    const failed = reduceAcnProcessState(Option.some(preparing), {
      _tag: "PreparationFailed",
      manager,
      reason: "download failed",
    })
    expect(failed.identityFloor).toBe("2.0.0")
    expect(failed.mode).toMatchObject({
      _tag: "Assigned",
      current: assigned.mode.current,
      result: { _tag: "Some", value: { _tag: "Failed", reason: "download failed" } },
    })
  })

  it("requires a takeover owner to prepare again before spawning", () => {
    const begun = reduceAcnProcessState(Option.none(), {
      _tag: "BeginEnsure",
      target: identity("1.0.0"),
      manager,
    })
    const prepared = reduceAcnProcessState(Option.some(begun), {
      _tag: "PreparationSucceeded",
      manager,
    })
    const replacementManager = {
      pid: 101,
      processStartIdentity: ProcessStartIdentitySchema.make("manager-101"),
    }
    const takenOver = reduceAcnProcessState(Option.some(prepared), {
      _tag: "TakeOver",
      manager: replacementManager,
    })
    expect(takenOver.mode).toMatchObject({
      _tag: "Changing",
      owner: { _tag: "Manager", process: replacementManager, phase: { _tag: "Preparing" } },
    })
    expect(() => reduceAcnProcessState(Option.some(takenOver), {
      _tag: "CandidateSpawned",
      manager: replacementManager,
      candidate,
    })).toThrow()
  })

  it("preserves one change identity while raising its identity floor", () => {
    const begun = reduceAcnProcessState(Option.none(), {
      _tag: "BeginEnsure",
      target: identity("1.0.0"),
      manager,
    })
    const upgraded = reduceAcnProcessState(Option.some(begun), {
      _tag: "UpgradeEnsure",
      target: identity("2.0.0"),
      manager: {
        pid: 101,
        processStartIdentity: ProcessStartIdentitySchema.make("manager-101"),
      },
    })
    expect(upgraded.identityFloor).toBe("2.0.0")
    expect(upgraded.mode._tag).toBe("Changing")
    if (begun.mode._tag !== "Changing" || upgraded.mode._tag !== "Changing") throw new Error("unreachable")
    expect(upgraded.mode.changeRevision).toBe(begun.mode.changeRevision)
    expect(upgraded.mode.purpose).toEqual({ _tag: "Ensure", target: "2.0.0" })
  })

  it("fences a candidate admission after takeover", () => {
    const begun = reduceAcnProcessState(Option.none(), {
      _tag: "BeginEnsure",
      target: identity("1.0.0"),
      manager,
    })
    const prepared = reduceAcnProcessState(Option.some(begun), {
      _tag: "PreparationSucceeded",
      manager,
    })
    const spawned = reduceAcnProcessState(Option.some(prepared), {
      _tag: "CandidateSpawned",
      manager,
      candidate,
    })
    const takenOver = reduceAcnProcessState(Option.some(spawned), {
      _tag: "TakeOver",
      manager: {
        pid: 101,
        processStartIdentity: ProcessStartIdentitySchema.make("manager-101"),
      },
    })
    expect(() => reduceAcnProcessState(Option.some(takenOver), {
      _tag: "CandidateAdmitted",
      candidate,
      id: "candidate" as never,
      url: "http://127.0.0.1:1234",
    })).toThrow()
  })

  it("does not turn an active termination into an ensure upgrade", () => {
    const spawned = spawnedCandidate()
    const assigned = reduceAcnProcessState(Option.some(spawned), {
      _tag: "CandidateAdmitted",
      candidate,
      id: "candidate" as never,
      url: "http://127.0.0.1:1234",
    })
    if (assigned.mode._tag !== "Assigned") throw new Error("unreachable")
    const terminating = reduceAcnProcessState(Option.some(assigned), {
      _tag: "BeginTerminate",
      manager,
      current: assigned.mode.current,
    })
    expect(() => reduceAcnProcessState(Option.some(terminating), {
      _tag: "UpgradeEnsure",
      target: identity("2.0.0"),
      manager,
    })).toThrow()
  })

  it("administratively terminates the exact assigned ACN", () => {
    const spawned = spawnedCandidate()
    const assigned = reduceAcnProcessState(Option.some(spawned), {
      _tag: "CandidateAdmitted",
      candidate,
      id: "candidate" as never,
      url: "http://127.0.0.1:1234",
    })
    const terminating = reduceAcnProcessState(Option.some(assigned), {
      _tag: "BeginTerminateCurrent",
      manager,
    })
    expect(terminating.mode).toMatchObject({
      _tag: "Changing",
      purpose: { _tag: "Terminate" },
      owner: { _tag: "Manager", phase: { _tag: "RetiringAssigned" } },
    })
  })

  it("cancels unpublished spawning without starting or killing a process", () => {
    const begun = reduceAcnProcessState(Option.none(), {
      _tag: "BeginEnsure",
      target: identity("1.0.0"),
      manager,
    })
    const stopped = reduceAcnProcessState(Option.some(begun), {
      _tag: "BeginTerminateCurrent",
      manager,
    })
    expect(stopped.mode._tag).toBe("Unassigned")
    if (stopped.mode._tag !== "Unassigned") throw new Error("unreachable")
    expect(Option.getOrThrow(stopped.mode.result)._tag).toBe("Terminated")
  })

  it("fences and retires an admitted candidate during administrative stop", () => {
    const terminating = reduceAcnProcessState(Option.some(spawnedCandidate()), {
      _tag: "BeginTerminateCurrent",
      manager,
    })
    expect(terminating.mode).toMatchObject({
      _tag: "Changing",
      purpose: { _tag: "Terminate" },
      owner: { _tag: "Manager", phase: { _tag: "RetiringCandidate", candidate } },
    })
    const stopped = reduceAcnProcessState(Option.some(terminating), {
      _tag: "CandidateExited",
      manager,
    })
    expect(stopped.mode._tag).toBe("Unassigned")
  })

  it("preserves exact blocked-candidate cleanup while transferring stop ownership", () => {
    const blockedOwner = {
      pid: 101,
      processStartIdentity: ProcessStartIdentitySchema.make("manager-101"),
    }
    const takenOver = reduceAcnProcessState(Option.some(spawnedCandidate()), {
      _tag: "TakeOver",
      manager: blockedOwner,
    })
    const blocked = reduceAcnProcessState(Option.some(takenOver), {
      _tag: "CandidateCleanupBlocked",
      manager: blockedOwner,
      reason: "still alive",
    })
    const terminating = reduceAcnProcessState(Option.some(blocked), {
      _tag: "BeginTerminateCurrent",
      manager,
    })
    expect(terminating.mode).toMatchObject({
      _tag: "Changing",
      purpose: { _tag: "Terminate" },
      owner: { _tag: "Manager", process: manager, phase: { _tag: "RetiringCandidate", candidate } },
    })
  })

  it("commits an independently dead candidate as failure instead of retrying", () => {
    const failed = reduceAcnProcessState(Option.some(spawnedCandidate()), {
      _tag: "CandidateFailed",
      candidate,
      reason: "candidate exited",
    })
    expect(failed.mode._tag).toBe("Unassigned")
    if (failed.mode._tag !== "Unassigned") throw new Error("unreachable")
    expect(Option.getOrThrow(failed.mode.result)).toMatchObject({
      _tag: "Failed",
      reason: "candidate exited",
    })
  })

  it("retains blocked candidate cleanup and requires an explicit retry", () => {
    const replacementManager = {
      pid: 101,
      processStartIdentity: ProcessStartIdentitySchema.make("manager-101"),
    }
    const takenOver = reduceAcnProcessState(Option.some(spawnedCandidate()), {
      _tag: "TakeOver",
      manager: replacementManager,
    })
    const blocked = reduceAcnProcessState(Option.some(takenOver), {
      _tag: "CandidateCleanupBlocked",
      manager: replacementManager,
      reason: "still alive",
    })
    expect(blocked.mode).toMatchObject({
      _tag: "Changing",
      owner: {
        _tag: "Manager",
        phase: { _tag: "BlockedCandidateCleanup", candidate, reason: "still alive" },
      },
    })
    const retrying = reduceAcnProcessState(Option.some(blocked), {
      _tag: "RetryCandidateCleanup",
      manager,
    })
    expect(retrying.mode).toMatchObject({
      _tag: "Changing",
      owner: { _tag: "Manager", process: manager, phase: { _tag: "RetiringCandidate", candidate } },
    })
  })

  it("admits exactly one concurrent writer for a revision", async () => {
    await run(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-state-" })
      const attempts = yield* Effect.all(
        Array.from({ length: 16 }, () => applyAcnProcessCommand({
          dataDirectory,
          expectedRevision: Option.none(),
          command: { _tag: "BeginEnsure", target: identity("1.0.0"), manager },
        }).pipe(Effect.either)),
        { concurrency: "unbounded" },
      )
      expect(attempts.filter(Either.isRight)).toHaveLength(1)
      expect(attempts.filter(Either.isLeft).every((attempt) =>
        Either.isLeft(attempt) && attempt.left instanceof AcnProcessStateConflict
      )).toBe(true)
      const state = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
      expect(state.revision).toBe(1)
    }))
  })

  it("does not treat a malformed highest revision as absence", async () => {
    await run(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-state-" })
      const first: AcnProcessState = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: identity("1.0.0"), manager },
      })
      const directory = `${dataDirectory}/acn/process-state`
      yield* fs.writeFileString(`${directory}/0000000000000002.json`, "not json")
      expect(first.revision).toBe(1)
      expect(Either.isLeft(yield* readAcnProcessState(dataDirectory).pipe(Effect.either))).toBe(true)
    }))
  })
})
