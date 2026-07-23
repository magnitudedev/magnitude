import { Context, Effect, Layer, Option, Ref, Stream } from "effect"
import {
  LocalModelMutationFailed,
  type LocalInferenceError,
} from "@magnitudedev/protocol"
import { IcnClient } from "@magnitudedev/icn"
import type { ProviderModelId } from "@magnitudedev/sdk"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { offeringTargetToIcn, servingProfileToIcn } from "./local-model-icn-adapter"

const failure = (operation: string, error: unknown) => {
  if (typeof error === "object"
    && error !== null
    && "_tag" in error
    && error._tag === "GeneratedClientRemoteError"
    && "body" in error
    && typeof error.body === "object"
    && error.body !== null
    && "error" in error.body
    && typeof error.body.error === "object"
    && error.body.error !== null) {
    const body = error.body.error as {
      readonly code?: unknown
      readonly message?: unknown
      readonly retryable?: unknown
    }
    return new LocalModelMutationFailed({
      code: typeof body.code === "string" ? body.code : operation,
      message: typeof body.message === "string" ? body.message : String(error),
      retryable: typeof body.retryable === "boolean" ? body.retryable : true,
    })
  }
  return new LocalModelMutationFailed({
    code: operation,
    message: error instanceof Error ? error.message : String(error),
    retryable: true,
  })
}

export interface LocalModelRuntimeApi {
  readonly load: (
    providerModelId: ProviderModelId,
    onProgress: (progress: LocalModelLoadProgress) => Effect.Effect<void>,
  ) => Effect.Effect<void, LocalInferenceError>
  readonly unload: (providerModelId: ProviderModelId) => Effect.Effect<void, LocalInferenceError>
  readonly isResident: (providerModelId: ProviderModelId) => Effect.Effect<boolean>
}

export interface LocalModelLoadProgress {
  readonly stage: "queued" | "resolving" | "unloading" | "loading" | "verifying"
  readonly fraction: Option.Option<number | null>
}

export class LocalModelRuntime extends Context.Tag("LocalModelRuntime")<
  LocalModelRuntime,
  LocalModelRuntimeApi
>() {}

export const LocalModelRuntimeLive: Layer.Layer<
  LocalModelRuntime,
  never,
  IcnClient | LocalProviderOfferings
> = Layer.effect(LocalModelRuntime, Effect.gen(function* () {
  const client = yield* IcnClient
  const offerings = yield* LocalProviderOfferings
  const resident = yield* Ref.make<OptionallyResident | undefined>(undefined)
  const lock = yield* Effect.makeSemaphore(1)

  return LocalModelRuntime.of({
    load: (providerModelId, onProgress) => lock.withPermits(1)(Effect.gen(function* () {
      const offering = yield* offerings.resolve(providerModelId)
      const target = yield* offeringTargetToIcn(offering.configuration.target).pipe(
        Effect.mapError((error) => failure("encode_local_model_target_failed", error)),
      )
      const response = yield* client.models.loadModelConfiguration({
        payload: {
          configuration: {
            id: offering.configuration.id,
            target,
            profile: servingProfileToIcn(offering.configuration.profile),
          },
        },
      }).pipe(
        Effect.mapError((error) => failure("load_local_model_failed", error)),
        Effect.tapError(() => Ref.set(resident, undefined)),
      )
      let ready = false
      yield* response.events.pipe(
        Stream.takeUntil((event) => event._tag !== "Progress"),
        Stream.runForEach((event) => {
          switch (event._tag) {
            case "Progress":
              return onProgress({ stage: event.stage, fraction: event.fraction })
            case "Failed":
              return Effect.fail(new LocalModelMutationFailed(event.failure))
            case "Ready":
              ready = true
              return Ref.set(resident, {
                providerModelId,
                residencyId: event.ready.residencyId,
              })
          }
        }),
        Effect.mapError((error) => error instanceof LocalModelMutationFailed
          ? error
          : failure("load_local_model_failed", error)),
        Effect.tapError(() => Ref.set(resident, undefined)),
      )
      if (!ready) {
        return yield* new LocalModelMutationFailed({
          code: "incomplete_load_stream",
          message: "ICN ended the model load stream before reporting a terminal result",
          retryable: true,
        })
      }
      // ICN admits exactly one residency. A successful load is authoritative and
      // replaces any prior residency, including one for another provider model.
    })),
    unload: (providerModelId) => lock.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(resident)
      if (!current || current.providerModelId !== providerModelId) return
      yield* client.models.unloadModelResidency({
        path: { residency_id: current.residencyId },
      }).pipe(
        Effect.catchAll((error) =>
          error._tag === "GeneratedClientRemoteError" && error.status === 404
            ? Effect.void
            : Effect.fail(failure("unload_local_model_failed", error))),
      )
      yield* Ref.set(resident, undefined)
    })),
    isResident: (providerModelId) => Ref.get(resident).pipe(
      Effect.map((current) => current?.providerModelId === providerModelId),
    ),
  })
}))

interface OptionallyResident {
  readonly providerModelId: ProviderModelId
  readonly residencyId: string
}
