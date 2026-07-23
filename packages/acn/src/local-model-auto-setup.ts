import { Context, Effect, Layer, PubSub, Ref, Stream } from "effect"
import {
  modelOfferingTargetPackageIds,
  type ModelFailure,
  type ModelOfferingTargetId,
} from "@magnitudedev/protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import { LocalModelEvaluations } from "./local-model-evaluations"
import { LocalModelPackages } from "./local-model-packages"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"

export type LocalModelAutoSetupStatus =
  | { readonly _tag: "Preparing" }
  | { readonly _tag: "Unavailable"; readonly failure: ModelFailure }

export interface LocalModelAutoSetupApi {
  readonly statuses: Effect.Effect<ReadonlyMap<ModelOfferingTargetId, LocalModelAutoSetupStatus>>
  readonly changes: Stream.Stream<void>
}

export class LocalModelAutoSetup extends Context.Tag("LocalModelAutoSetup")<
  LocalModelAutoSetup,
  LocalModelAutoSetupApi
>() {}

/**
 * Creates one automatic offering for each usable standalone package discovered
 * on disk. Existing offerings are assessed by their consumers and are never
 * silently replaced or refitted here.
 */
export const LocalModelAutoSetupLive: Layer.Layer<
  LocalModelAutoSetup,
  never,
  IcnCatalog | IcnHardware | LocalModelEvaluations | LocalModelPackages | LocalProviderOfferings
> = Layer.scoped(LocalModelAutoSetup, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const hardware = yield* IcnHardware
  const packages = yield* LocalModelPackages
  const evaluations = yield* LocalModelEvaluations
  const offerings = yield* LocalProviderOfferings
  const attempted = yield* Ref.make<ReadonlySet<string>>(new Set())
  const statuses = yield* Ref.make<ReadonlyMap<ModelOfferingTargetId, LocalModelAutoSetupStatus>>(
    new Map(),
  )
  const changes = yield* PubSub.sliding<void>(16)
  const lock = yield* Effect.makeSemaphore(1)

  const reconcile = lock.withPermits(1)(Effect.gen(function* () {
    if (!(yield* catalog.ready)) return
    const configured = yield* offerings.list
    const topology = (yield* hardware.get).state.topology_fingerprint
    const configuredPackages = new Set(configured.flatMap(({ configuration }) =>
      modelOfferingTargetPackageIds(configuration.target)))
    const catalogTargets = yield* Effect.forEach(
      (yield* catalog.get).state.models,
      recommendableModelFromIcn,
    )
    const explicitStandalonePackageIds = new Set(catalogTargets.flatMap(({ target }) =>
      target._tag === "Package" ? [target.package.id] : []))
    const speculativePackageIds = new Set(catalogTargets.flatMap(({ target }) =>
      target._tag === "SpeculativeDecodingPair" ? [target.target.id, target.draft.id] : []))
    const attemptedPackages = yield* Ref.get(attempted)
    const candidates = (yield* packages.snapshot).state.entries.filter((entry) =>
      entry.localState._tag === "Installed"
      && entry.inspection._tag === "Inspected"
      && (!speculativePackageIds.has(entry.package.id)
        || explicitStandalonePackageIds.has(entry.package.id))
      && !configuredPackages.has(entry.package.id)
      && !attemptedPackages.has(`${topology}:${entry.package.id}`))

    for (const candidate of candidates) {
      const attemptKey = `${topology}:${candidate.package.id}`
      if (candidate.targetId._tag === "None") continue
      const modelId = candidate.targetId.value
      yield* Ref.update(statuses, (current) =>
        new Map(current).set(modelId, { _tag: "Preparing" }))
      yield* PubSub.publish(changes, undefined)
      const result = yield* evaluations.fit({ _tag: "Package", package: candidate.package }).pipe(
        Effect.flatMap(({ modelId, configuration }) =>
          offerings.save(modelId, configuration, { _tag: "Automatic" })),
        Effect.tapError((error) => Effect.logWarning("Unable to auto-fit installed model package").pipe(
          Effect.annotateLogs({ packageId: candidate.package.id, cause: error.message }),
        )),
        Effect.either,
      )
      if (result._tag === "Right"
        || ("retryable" in result.left && !result.left.retryable)) {
        yield* Ref.update(attempted, (current) => new Set([...current, attemptKey]))
      }
      yield* Ref.update(statuses, (current) => {
        const next = new Map(current)
        if (result._tag === "Right") {
          next.delete(modelId)
        } else {
          next.set(modelId, {
            _tag: "Unavailable",
            failure: {
              code: "code" in result.left ? result.left.code : result.left._tag,
              message: result.left.message,
              retryable: "retryable" in result.left ? result.left.retryable : false,
            },
          })
        }
        return next
      })
      yield* PubSub.publish(changes, undefined)
    }
  })).pipe(Effect.catchAllCause((cause) =>
    Effect.logWarning("Unable to reconcile installed local models").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )))

  yield* reconcile
  yield* Stream.merge(
    Stream.merge(packages.changes, hardware.changes),
    catalog.changes,
  ).pipe(
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )
  return LocalModelAutoSetup.of({
    statuses: Ref.get(statuses),
    changes: Stream.fromPubSub(changes),
  })
}))
