import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import {
  ProcessGroupAbsent,
  ProcessGroupPresent,
  type AcnOwnerRecord,
  type AcnOwnerStore,
  type ExactProcess,
  type ProcessGroupController,
} from "@magnitudedev/acn-protocol/coordination"
import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  AcnProcessAttestationFailed,
  attestAcnProcessTreeExitWith,
  captureAcnProcessAttestationWith,
  type AcnProcessAttestationServices,
} from "./acn-process-attestation"

const original: AcnOwnerRecord = {
  pid: 101,
  processStartIdentity: ProcessStartIdentitySchema.make("original"),
  port: 4_001,
}
const successor: AcnOwnerRecord = {
  pid: 202,
  processStartIdentity: ProcessStartIdentitySchema.make("successor"),
  port: 4_002,
}

const services = (options: {
  readonly current?: AcnOwnerRecord
  readonly identities?: ReadonlyMap<number, ExactProcess["processStartIdentity"]>
  readonly absent?: ReadonlySet<number>
} = {}): AcnProcessAttestationServices => {
  const current = options.current ?? original
  const owners: AcnOwnerStore = {
    current: Effect.succeed(Option.some(current)),
    replaceOwner: () => Effect.die("unused"),
  }
  const processes: ProcessGroupController = {
    inspect: (pid) => Effect.succeed(Option.fromNullable(options.identities?.get(pid))),
    currentProcess: Effect.die("unused"),
    observeGroup: (group) => Effect.succeed(
      options.absent?.has(group.leader.pid)
        ? new ProcessGroupAbsent({ group })
        : new ProcessGroupPresent({ group }),
    ),
    signalGroup: () => Effect.die("unused"),
  }
  return { owners, processes }
}

describe("ACN process attestation", () => {
  it("captures only the exact recorded owner occurrence", async () => {
    const captured = await Effect.runPromise(captureAcnProcessAttestationWith(services({
      identities: new Map([[original.pid, original.processStartIdentity]]),
    })))
    expect(captured.owner).toEqual(original)

    const mismatch = await Effect.runPromiseExit(captureAcnProcessAttestationWith(services({
      identities: new Map([[original.pid, ProcessStartIdentitySchema.make("reused")]]),
    })))
    expect(mismatch._tag).toBe("Failure")
  })

  it("fails when the original process group survives", async () => {
    const exit = await Effect.runPromiseExit(attestAcnProcessTreeExitWith(
      services({ current: original, absent: new Set() }),
      { owner: original },
      0,
    ))
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure" && exit.cause._tag === "Fail") {
      expect(exit.cause.error).toBeInstanceOf(AcnProcessAttestationFailed)
      expect(exit.cause.error.message).toContain("remained present")
    }
  })

  it("fails when coordination identifies a live successor owner", async () => {
    const exit = await Effect.runPromiseExit(attestAcnProcessTreeExitWith(
      services({
        current: successor,
        identities: new Map([[successor.pid, successor.processStartIdentity]]),
        absent: new Set([original.pid]),
      }),
      { owner: original },
      0,
    ))
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure" && exit.cause._tag === "Fail") {
      expect(exit.cause.error.message).toContain("live owner")
    }
  })

  it("accepts an absent exact tree with only a stale owner row", async () => {
    await Effect.runPromise(attestAcnProcessTreeExitWith(
      services({ current: original, absent: new Set([original.pid]) }),
      { owner: original },
      0,
    ))
  })
})
