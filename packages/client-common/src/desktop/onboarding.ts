import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { CatalogFormModelIdSchema, type CatalogLocalModel } from "@magnitudedev/sdk"
import { FSM } from "@magnitudedev/utils"
import { LocalModels, localModelFailureMessage } from "../local-models/service"
import { modelDownloadFailureMessage } from "../local-models/failure-messages"
import { OnboardingPersistence } from "../onboarding/persistence"

class Idle extends Schema.TaggedClass<Idle>()("Idle", {}) {}
class Installing extends Schema.TaggedClass<Installing>()("Installing", { modelId: CatalogFormModelIdSchema }) {}
class Loading extends Schema.TaggedClass<Loading>()("Loading", { modelId: CatalogFormModelIdSchema }) {}
class Ready extends Schema.TaggedClass<Ready>()("Ready", { modelId: CatalogFormModelIdSchema }) {}
class Failed extends Schema.TaggedClass<Failed>()("Failed", { modelId: CatalogFormModelIdSchema, message: Schema.String }) {}
const states = { Idle, Installing, Loading, Ready, Failed }
export type DesktopSetupState = Idle | Installing | Loading | Ready | Failed
const machine = FSM.defineFSM(states, {
  Idle: ["Installing"], Installing: ["Loading", "Failed"], Loading: ["Ready", "Failed"],
  Ready: ["Installing", "Idle"], Failed: ["Installing", "Idle"],
} as const)
class SetupUnavailable extends Schema.TaggedError<SetupUnavailable>()("SetupUnavailable", { message: Schema.String }) {}

/** Completion predicates consume the catalog projection, never mutation history. */
export const desktopSetupModelOutcome = (model: Pick<CatalogLocalModel, "acquisitionState">, phase: "Installing" | "Loading"):
  { readonly _tag: "Waiting" | "Ready" } | { readonly _tag: "Failed"; readonly message: string } => {
  const acquisition = model.acquisitionState
  if (acquisition._tag === "InstallFailed" || acquisition._tag === "UpdateFailed") {
    return { _tag: "Failed", message: modelDownloadFailureMessage(acquisition.failure) }
  }
  if (phase === "Installing") {
    if (acquisition._tag === "Installing" || acquisition._tag === "Updating") return { _tag: "Waiting" }
    if (acquisition._tag === "Installed") return { _tag: "Ready" }
    return { _tag: "Failed", message: "The download did not finish. Choose the model again to retry." }
  }
  if (!("residencyState" in acquisition)) return { _tag: "Failed", message: "The selected model is no longer installed." }
  const residency = acquisition.residencyState
  if (residency._tag === "Ready") return { _tag: "Ready" }
  if (residency._tag === "Requested" || residency._tag === "Loading") return { _tag: "Waiting" }
  return { _tag: "Failed", message: residency._tag === "Failed" ? residency.failure.message : "Model loading was stopped. You can try again." }
}

const makeDesktopOnboarding = Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const models = yield* LocalModels
  const persistence = yield* OnboardingPersistence
  const scope = yield* Effect.scope
  const admission = yield* Effect.makeSemaphore(1)
  const state = Atom.keepAlive(Atom.make<DesktopSetupState>(new Idle({})))
  const busy = () => { const value = registry.get(state); return value._tag === "Installing" || value._tag === "Loading" }
  const awaitModel = (modelId: typeof CatalogFormModelIdSchema.Type, phase: "Installing" | "Loading") =>
    Registry.toStream(registry, models.state).pipe(
      Stream.filter(result => !Result.isInitial(result)),
      Stream.mapEffect(result => Result.isFailure(result) ? Effect.fail(new SetupUnavailable({ message: "Model observation is unavailable. Check Status before trying setup again." })) : Effect.succeed(result)),
      Stream.filter(Result.isSuccess),
      Stream.map(result => {
        const model = result.value.models.find(model => model.modelId === modelId && model._tag === "Catalog")
        return model?._tag === "Catalog" ? desktopSetupModelOutcome(model, phase)
          : { _tag: "Failed" as const, message: "The selected model is no longer in the catalog." }
      }),
      Stream.filter(outcome => outcome._tag !== "Waiting"), Stream.runHead,
      Effect.flatMap(outcome => Option.isSome(outcome) && outcome.value._tag === "Ready" ? Effect.void
        : Effect.fail(new SetupUnavailable({ message: Option.isSome(outcome) && outcome.value._tag === "Failed"
          ? outcome.value.message : "Model observation ended before setup finished." }))),
    )
  const select = (modelId: typeof CatalogFormModelIdSchema.Type) => admission.withPermits(1)(Effect.uninterruptibleMask(restore => Effect.gen(function* () {
    const previous = registry.get(state)
    if (previous._tag === "Installing" || previous._tag === "Loading") return
    const installing = machine.transition(previous, "Installing", { modelId })
    registry.set(state, installing)
    yield* restore(Effect.gen(function* () {
      yield* models.install(modelId)
      yield* awaitModel(modelId, "Installing")
      const loading = machine.transition(installing, "Loading", {})
      registry.set(state, loading)
      yield* models.load(modelId)
      yield* awaitModel(modelId, "Loading")
      registry.set(state, machine.transition(loading, "Ready", {}))
    })).pipe(
      Effect.catchAllCause(cause => Effect.sync(() => {
        const current = registry.get(state)
        if (current._tag !== "Installing" && current._tag !== "Loading") return
        const message = localModelFailureMessage(cause, "Setup could not finish. Check Status and try again.")
        registry.set(state, machine.transition(current, "Failed", { message }))
      })),
      Effect.provideService(Registry.AtomRegistry, registry), Effect.forkIn(scope),
    )
  })))
  const finish = admission.withPermits(1)(Effect.gen(function* () {
    if (busy()) return yield* new SetupUnavailable({ message: "Wait for setup to finish, or cancel the model operation first." })
    yield* persistence.complete
    const current = registry.get(state)
    if (current._tag === "Ready" || current._tag === "Failed") registry.set(state, machine.transition(current, "Idle", {}))
  }))
  return { state: state as Atom.Atom<DesktopSetupState>, completion: persistence.state, retry: persistence.retry, select, finish }
})
export interface DesktopOnboarding extends Effect.Effect.Success<typeof makeDesktopOnboarding> {}
export const DesktopOnboarding = Context.GenericTag<DesktopOnboarding>("client/DesktopOnboarding")
export const DesktopOnboardingLive = Layer.scoped(DesktopOnboarding, makeDesktopOnboarding)
