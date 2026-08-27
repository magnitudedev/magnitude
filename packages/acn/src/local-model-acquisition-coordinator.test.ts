import { describe, expect, it } from "@effect/vitest"
import { Effect, Option } from "effect"
import { ModelDownloadIdSchema } from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import {
  LocalModelAcquisitionCoordinator,
  LocalModelAcquisitionCoordinatorLive,
} from "./local-model-acquisition-coordinator"

const modelA = ProviderModelIdSchema.make("model-a")
const modelB = ProviderModelIdSchema.make("model-b")
const downloadA = ModelDownloadIdSchema.make("download-a")

describe("LocalModelAcquisitionCoordinator", () => {
  it.effect("admits one synchronization per model while allowing unrelated models", () =>
    Effect.gen(function* () {
      const coordinator = yield* LocalModelAcquisitionCoordinator
      const first = yield* coordinator.admitSync(modelA, Option.none())
      const duplicate = yield* coordinator.admitSync(modelA, Option.none())
      const unrelated = yield* coordinator.admitSync(modelB, Option.none())

      expect(Option.isSome(first)).toBe(true)
      expect(Option.isNone(duplicate)).toBe(true)
      expect(Option.isSome(unrelated)).toBe(true)
    }).pipe(Effect.provide(LocalModelAcquisitionCoordinatorLive)))

  it.effect("retains cancellation accepted before native download correlation", () =>
    Effect.gen(function* () {
      const coordinator = yield* LocalModelAcquisitionCoordinator
      const generation = Option.getOrThrow(yield* coordinator.admitSync(modelA, Option.none()))

      expect(Option.isNone(yield* coordinator.requestSyncCancellation(modelA))).toBe(true)
      expect(Option.getOrThrow(yield* coordinator.correlateSync(modelA, generation, downloadA))).toBe(true)
      expect(Option.getOrThrow(yield* coordinator.requestSyncCancellation(modelA))).toEqual({
        generation,
        downloadId: downloadA,
      })
    }).pipe(Effect.provide(LocalModelAcquisitionCoordinatorLive)))

  it.effect("discards stale completion from an older generation", () =>
    Effect.gen(function* () {
      const coordinator = yield* LocalModelAcquisitionCoordinator
      const first = Option.getOrThrow(yield* coordinator.admitSync(modelA, Option.none()))
      yield* coordinator.failSyncAdmission(modelA, first, {
        _tag: "Internal",
        message: "first failed",
      })
      const second = Option.getOrThrow(yield* coordinator.admitSync(modelA, Option.some(first)))

      yield* coordinator.finishSync(modelA, first)
      expect(Option.isNone(yield* coordinator.admitSync(modelA, Option.some(first)))).toBe(true)
      expect((yield* coordinator.state).syncs.get(modelA)?.generation).toBe(second)
    }).pipe(Effect.provide(LocalModelAcquisitionCoordinatorLive)))

  it.effect("rejects stale native correlation instead of orphaning it as an accepted sync", () =>
    Effect.gen(function* () {
      const coordinator = yield* LocalModelAcquisitionCoordinator
      const first = Option.getOrThrow(yield* coordinator.admitSync(modelA, Option.none()))
      yield* coordinator.failSyncAdmission(modelA, first, {
        _tag: "Internal",
        message: "first failed",
      })
      const second = Option.getOrThrow(yield* coordinator.admitSync(modelA, Option.some(first)))

      expect(Option.isNone(yield* coordinator.correlateSync(modelA, first, downloadA))).toBe(true)
      expect((yield* coordinator.state).syncs.get(modelA)?.generation).toBe(second)
    }).pipe(Effect.provide(LocalModelAcquisitionCoordinatorLive)))

  it.effect("serializes synchronization and removal for the same model", () =>
    Effect.gen(function* () {
      const coordinator = yield* LocalModelAcquisitionCoordinator
      const synchronization = Option.getOrThrow(yield* coordinator.admitSync(modelA, Option.none()))
      expect(Option.isNone(yield* coordinator.admitRemoval(modelA))).toBe(true)

      yield* coordinator.finishSync(modelA, synchronization)
      const removal = Option.getOrThrow(yield* coordinator.admitRemoval(modelA))
      expect(Option.isNone(yield* coordinator.admitSync(modelA, Option.none()))).toBe(true)

      yield* coordinator.finishRemoval(modelA, removal)
      expect(Option.isSome(yield* coordinator.admitSync(modelA, Option.none()))).toBe(true)
    }).pipe(Effect.provide(LocalModelAcquisitionCoordinatorLive)))

  it.effect("supersedes terminal failure presentation when the opposite command is admitted", () =>
    Effect.gen(function* () {
      const coordinator = yield* LocalModelAcquisitionCoordinator
      const synchronization = Option.getOrThrow(yield* coordinator.admitSync(modelA, Option.none()))
      yield* coordinator.failSyncAdmission(modelA, synchronization, {
        _tag: "Internal",
        message: "sync failed",
      })

      const removal = Option.getOrThrow(yield* coordinator.admitRemoval(modelA))
      expect((yield* coordinator.state).syncs.has(modelA)).toBe(false)
      yield* coordinator.failRemoval(modelA, removal, {
        code: "remove_failed",
        message: "remove failed",
        retryable: true,
      })

      expect(Option.isSome(yield* coordinator.admitSync(modelA, Option.none()))).toBe(true)
      expect((yield* coordinator.state).removals.has(modelA)).toBe(false)
    }).pipe(Effect.provide(LocalModelAcquisitionCoordinatorLive)))
})
