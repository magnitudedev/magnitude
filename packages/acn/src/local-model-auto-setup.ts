import { Context, Effect, Layer, Option, PubSub, Ref, Stream } from "effect"
import {
  LocalModelMutationFailed,
  modelOfferingTargetPackageIds,
  type ModelFailure,
  type ModelOfferingTargetId,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import {
  LocalModelAssessments,
  modelAssessmentProfiles,
  selectModelServingConfiguration,
} from "./local-model-assessments"
import { LocalModelPackages } from "./local-model-packages"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { recommendableModelFromIcn } from "./local-model-icn-adapter"

export interface LocalModelAutoSetupApi {
  readonly failures: Effect.Effect<ReadonlyMap<ModelOfferingTargetId, ModelFailure>>
  readonly changes: Stream.Stream<void>
}

export class LocalModelAutoSetup extends Context.Tag("LocalModelAutoSetup")<
  LocalModelAutoSetup,
  LocalModelAutoSetupApi
>() {}

/**
 * Creates one automatic offering for each usable standalone package discovered
 * on disk. Existing offerings are assessed by their consumers and are never
 * silently replaced here.
 */
export const LocalModelAutoSetupLive: Layer.Layer<
  LocalModelAutoSetup,
  never,
  IcnCatalog | IcnHardware | LocalModelAssessments | LocalModelPackages | LocalProviderOfferings
> = Layer.scoped(LocalModelAutoSetup, Effect.gen(function* () {
  const catalog = yield* IcnCatalog
  const hardware = yield* IcnHardware
  const packages = yield* LocalModelPackages
  const assessments = yield* LocalModelAssessments
  const offerings = yield* LocalProviderOfferings
  const attempted = yield* Ref.make<ReadonlySet<string>>(new Set())
  const failures = yield* Ref.make<ReadonlyMap<ModelOfferingTargetId, ModelFailure>>(
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

    yield* Effect.forEach(candidates, (candidate) => Effect.gen(function* () {
      const attemptKey = `${topology}:${candidate.package.id}`
      if (candidate.targetId._tag === "None") return
      const targetId = candidate.targetId.value
      const target = { _tag: "Package" as const, package: candidate.package }
      const result = yield* assessments.assess([{
        targetId,
        target,
        profiles: modelAssessmentProfiles(target),
      }], () => Effect.void).pipe(
        Effect.flatMap((assessmentResults) => Effect.gen(function* () {
          const assessmentResult = Option.fromNullable(assessmentResults[0])
          if (Option.isNone(assessmentResult)) {
            return yield* Effect.dieMessage("ICN returned no assessment result")
          }
          if (assessmentResult.value._tag === "InvalidTarget") {
            return yield* new LocalModelMutationFailed({
              code: "model_target_invalid",
              message: assessmentResult.value.message,
              retryable: false,
            })
          }
          const configuration = selectModelServingConfiguration(target, assessmentResult.value)
          if (Option.isNone(configuration)) {
            return yield* new LocalModelMutationFailed({
              code: "model_does_not_fit",
              message: "No assessed context length fits the available hardware capacity",
              retryable: false,
            })
          }
          return yield* offerings.save(
            assessmentResult.value.targetId,
            configuration.value,
            { _tag: "Automatic" },
          )
        })),
        Effect.tapError((error) => Effect.logWarning("Unable to assess installed model package").pipe(
          Effect.annotateLogs({ packageId: candidate.package.id, cause: error.message }),
        )),
        Effect.either,
      )
      if (result._tag === "Right"
        || ("retryable" in result.left && !result.left.retryable)) {
        yield* Ref.update(attempted, (current) => new Set([...current, attemptKey]))
      }
      yield* Ref.update(failures, (current) => {
        const next = new Map(current)
        if (result._tag === "Right") {
          next.delete(targetId)
        } else {
          next.set(targetId, {
            code: "code" in result.left ? result.left.code : result.left._tag,
            message: result.left.message,
            retryable: "retryable" in result.left ? result.left.retryable : false,
          })
        }
        return next
      })
      yield* PubSub.publish(changes, undefined)
    }), { concurrency: 8 })
  })).pipe(Effect.catchAllCause((cause) =>
    Effect.logWarning("Unable to reconcile installed local models").pipe(
      Effect.annotateLogs({ cause: String(cause) }),
    )))

  yield* Stream.make(undefined).pipe(
    Stream.concat(Stream.merge(
      Stream.merge(packages.changes, hardware.assessmentChanges),
      catalog.changes,
    )),
    Stream.runForEach(() => reconcile),
    Effect.forkScoped,
  )
  return LocalModelAutoSetup.of({
    failures: Ref.get(failures),
    changes: Stream.fromPubSub(changes),
  })
}))
