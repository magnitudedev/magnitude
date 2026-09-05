import {
  AcnInstanceIdSchema,
  AcnReady,
  AcnRevisionSchema,
  AcnStarting,
  ProcessStartIdentitySchema,
} from "@magnitudedev/acn-protocol"

import { Option } from "effect"

import { describe, expect, it } from "vitest"

import { DAEMON_TARGET, DAEMON_VERSION } from "../version"

import { AcnCandidateNotLaunched } from "./acn-candidate-launch-supervisor"

import { decideAcnConvergence, type AcnConvergenceSnapshot } from "./acn-convergence-decider"

import {
  AcnRecordedOwnerAbsent,
  AcnRecordedOwnerLiveWithHealth,
  type AcnHealthObservation,
} from "./acn-owner-observer"


const owner = {
  pid: 41_003,
  processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:owner"),
  port: 49_152,
}

const base = (observation: AcnConvergenceSnapshot["observation"]): AcnConvergenceSnapshot => ({
  target: DAEMON_TARGET,
  observation,
  candidate: new AcnCandidateNotLaunched({}),
  launchPrepared: false,
  now: 1_000,
  ownerObservedAt: 1_000,
  healthStateObservedAt: 1_000,
})

const health = (revision: number, state: AcnReady | AcnStarting): AcnHealthObservation => ({
  status: state._tag === "Ready" ? 200 : 503,
  health: {
    service: "magnitude-acn",
    version: DAEMON_VERSION,
    revision: AcnRevisionSchema.make(revision),
    id: AcnInstanceIdSchema.make("owner"),
    pid: owner.pid,
    state,
  },
})


describe("decideAcnConvergence", () => {
  it("launches exactly when there is no live owner and no prior candidate", () => {
    expect(decideAcnConvergence(base(
      new AcnRecordedOwnerAbsent({ expectedOwner: Option.none() }),
    ))).toEqual({ _tag: "LaunchCandidate" })
  })

  it("prepares launch material before asking to shut down a lower revision", () => {
    const observation = new AcnRecordedOwnerLiveWithHealth({
      owner,
      health: health(DAEMON_TARGET.revision - 1, new AcnReady({})),
    })
    expect(decideAcnConvergence(base(observation))).toEqual({ _tag: "PrepareLaunch" })
    expect(decideAcnConvergence({ ...base(observation), launchPrepared: true })).toMatchObject({
      _tag: "ShutdownDaemon",
      reason: "RevisionTooOld",
    })
  })

  it("does not reinterpret stable Starting activity as failed liveness", () => {
    const observation = new AcnRecordedOwnerLiveWithHealth({
      owner,
      health: health(DAEMON_TARGET.revision, new AcnStarting({
        activity: "Resolving",
        progress: Option.none(),
      })),
    })
    expect(decideAcnConvergence({
      ...base(observation),
      now: 45_000,
      ownerObservedAt: 1_000,
      healthStateObservedAt: 1_000,
    })).toEqual({ _tag: "Wait" })
  })
})
