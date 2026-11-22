import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option, Stream } from "effect"
import { Mutation } from "@magnitudedev/effect-query"
import { localModelIsOwnedByMagnitude, localModelResidency, ModelLoadStageSchema, type LocalModelsState, type ModelId, type ModelResidency } from "@magnitudedev/sdk"
import type { AppearancePreference } from "@magnitudedev/sdk/desktop-host"
import { LocalModels } from "../local-models/service"
import { ClientEffectQuery } from "../state/client-effect-query"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import { formatModelLoadPercentage, formatModelLoadStage, formatModelMemory, isMeasuredModelLoadStage } from "../utils/model-load"
import { ApplicationHost, ApplicationHostFailed } from "./host"
import { ApplicationRouter } from "./router"
import { landingPage, readyToChat } from "./landing"
import type { AddModelRequest, ApplicationPage, HostNotice, QuitFailureDecision } from "./contracts"
import { ModelTrayPresentation, ModelTrayStatus } from "./contracts"

export { AddModelRequest, ApplicationPage, type MainPage, HostAction, HostNotice, QuitFailureDecision, ModelTrayPresentation, ModelTrayStatus } from "./contracts"

export const activeLocalModel = (models: LocalModelsState) => {
  for (const model of models.models) {
    const residency = Option.getOrUndefined(localModelResidency(model))
    if (residency && residency._tag !== "Unloaded" && residency._tag !== "Stopped" && residency._tag !== "Failed") {
      return Option.some({ model, residency })
    }
  }
  return Option.none()
}
/** Every phase word the tray's model row shows, so it can hold room for the widest. */
export const MODEL_TRAY_PHASES: ReadonlyArray<string> = [
  ...ModelLoadStageSchema.literals.map(formatModelLoadStage),
  "Loaded",
  "Stopping",
]
const modelTrayStatus = (
  model: string,
  residency: Exclude<ModelResidency, { readonly _tag: "Unloaded" | "Stopped" | "Failed" }>,
): typeof ModelTrayStatus.Type => {
  switch (residency._tag) {
    case "Requested": return { model, phase: formatModelLoadStage("preparing"), detail: { _tag: "Working" } }
    case "Loading": return {
      model,
      phase: formatModelLoadStage(residency.stage),
      detail: isMeasuredModelLoadStage(residency.stage)
        ? { _tag: "Progress", fraction: residency.fraction }
        : { _tag: "Working" },
    }
    case "Ready": return { model, phase: "Loaded", detail: { _tag: "Memory", text: formatModelMemory(residency.allocation) } }
    case "Stopping": return { model, phase: "Stopping", detail: { _tag: "Working" } }
  }
}
const modelTrayLabel = (status: typeof ModelTrayStatus.Type): string => {
  switch (status.detail._tag) {
    case "Working": return `${status.model} · ${status.phase}`
    case "Progress": return `${status.model} · ${status.phase} ${formatModelLoadPercentage(status.detail.fraction)}`
    case "Memory": return `${status.model} · ${status.phase} · ${status.detail.text}`
  }
}
export const modelTrayPresentation = (models: LocalModelsState): typeof ModelTrayPresentation.Type => {
  const active = activeLocalModel(models)
  if (Option.isSome(active)) {
    const status = modelTrayStatus(formatLocalModelDisplayName(active.value.model), active.value.residency)
    return { label: modelTrayLabel(status), status: Option.some(status), canStop: true }
  }
  return {
    label: models.models.length === 0 && !models.preparation.assessment.complete ? "Reading model status…" : "No model loaded",
    status: Option.none(),
    canStop: false,
  }
}

const unavailable = new ApplicationHostFailed({ message: "This action isn't available here." })

const makeApplicationSession = Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const host = yield* ApplicationHost
  const router = yield* ApplicationRouter
  const models = yield* LocalModels
  const effectQuery = yield* ClientEffectQuery
  const provideRegistry = Effect.provideService(Registry.AtomRegistry, registry)
  const onboarding = effectQuery.Configuration.GetOnboarding({})
  const hasMagnitudeOwnedModel = (result: Result.Result<LocalModelsState, unknown>) =>
    Result.isSuccess(result) && result.value.models.some(localModelIsOwnedByMagnitude)
  // Whether onboarding opens over the page: none until the service answers, and false when it can't be read
  // or when Magnitude already owns a downloaded model on this computer (such as an upgrade from 0.2.17).
  const onboardingPending = Atom.keepAlive(Atom.make((get): Option.Option<boolean> => {
    const state = get(onboarding).result
    if (Result.isInitial(state)) return Option.none()
    if (!Result.isSuccess(state) || state.value.onboarding !== "Pending") return Option.some(false)
    return Option.some(!hasMagnitudeOwnedModel(get(models.state)))
  }))
  // If GetOnboarding answered Pending before initial model discovery reported an existing Magnitude download,
  // persist completion once both onboarding state and the owned model have been observed.
  const onboardingNeedsAutoComplete = Atom.keepAlive(Atom.make((get): boolean => {
    const state = get(onboarding).result
    return Result.isSuccess(state) && state.value.onboarding === "Pending" && hasMagnitudeOwnedModel(get(models.state))
  }))
  yield* Registry.toStream(registry, onboardingNeedsAutoComplete).pipe(
    Stream.filter(Boolean),
    Stream.take(1),
    Stream.runForEach(() => Mutation.execute(effectQuery.Configuration.CompleteOnboarding, {}).pipe(provideRegistry, Effect.ignore)),
    Effect.forkScoped,
  )
  const navigate = router.navigate
  const landing = (result: Result.Result<LocalModelsState, unknown>) => Result.isSuccess(result) ? landingPage(result.value) : Option.none()
  // The app opens on the page the first full answer about this computer's models calls for, unless it has moved on.
  yield* Registry.toStream(registry, models.state).pipe(
    Stream.filterMap(landing),
    Stream.take(1),
    Stream.runForEach(router.land),
    Effect.forkScoped,
  )
  // The model onboarding's download fetches: once it is ready it is loaded and Chat opens, so onboarding ends in a chat.
  const onboardingDownload = Atom.keepAlive(Atom.make(Option.none<ModelId>()))
  const completeOnboarding = Atom.fn((download: Option.Option<ModelId>) => Mutation.execute(effectQuery.Configuration.CompleteOnboarding, {}).pipe(
    provideRegistry,
    Effect.zipRight(Effect.sync(() => registry.set(onboardingDownload, download))),
    Effect.zipRight(Effect.suspend(() => navigate(Option.getOrElse(landing(registry.get(models.state)), (): ApplicationPage => "chat")))),
  ))
  yield* Registry.toStream(registry, onboardingDownload).pipe(
    Stream.filterMap((download) => download),
    Stream.flatMap((modelId) => Registry.toStream(registry, models.state).pipe(
      Stream.filterMap((result): Option.Option<boolean> => {
        if (!Result.isSuccess(result)) return Option.none()
        const model = result.value.models.find((candidate) => candidate.modelId === modelId)
        // Ready, it is loaded; gone (cancelled) or failed, onboarding has nothing more to do.
        return model === undefined || model.acquisitionState._tag === "InstallFailed" ? Option.some(false)
          : readyToChat(model) ? Option.some(true) : Option.none()
      }),
      Stream.take(1),
      Stream.mapEffect((ready) => Effect.sync(() => registry.set(onboardingDownload, Option.none())).pipe(
        Effect.zipRight(ready ? models.load(modelId).pipe(Effect.zipRight(navigate("chat")), Effect.catchAll(Effect.logError)) : Effect.void))),
    ), { switch: true }),
    Stream.runDrain,
    Effect.forkScoped,
  )
  const notices = Atom.keepAlive(Atom.make<ReadonlyArray<HostNotice>>([]))
  const dismissNotice = (notice: HostNotice) => Effect.sync(() => registry.set(notices, registry.get(notices).filter(entry => entry !== notice)))
  const quitFailed = Atom.keepAlive(Atom.make(false))
  // A request to add a model belongs to the visit to Models it opened, so opening Models again drops it.
  const addModelRequested = Atom.keepAlive(Atom.make(Option.none<{ readonly visit: number; readonly request: AddModelRequest }>()))
  const addModelRequest = Atom.keepAlive(Atom.make(get => Option.flatMap(get(addModelRequested),
    entry => entry.visit === get(router.visit) ? Option.some(entry.request) : Option.none())))
  const requestAddModel = (request: AddModelRequest) => navigate("models").pipe(
    Effect.zipRight(Effect.sync(() => registry.set(addModelRequested, Option.some({ visit: registry.get(router.visit), request })))))
  const clearAddModelRequest = Effect.sync(() => registry.set(addModelRequested, Option.none()))

  if (Option.isSome(host.shell)) {
    const shell = host.shell.value
    yield* shell.actions.pipe(
      Stream.runForEach(action => {
        switch (action._tag) {
          case "Navigate": return navigate(action.page)
          case "StopModel": return models.stop.pipe(Effect.asVoid, Effect.catchAll(Effect.logError))
          case "ShowNotice": return Effect.sync(() => registry.set(notices, [...registry.get(notices), action.notice]))
          case "QuitFailed": return Effect.sync(() => registry.set(quitFailed, true))
        }
      }),
      Effect.catchAll(Effect.logError),
      Effect.forkScoped,
    )
    yield* Registry.toStream(registry, models.state).pipe(
      Stream.map(result => Result.isSuccess(result) ? modelTrayPresentation(result.value) : { label: "Model status unavailable", status: Option.none(), canStop: false }),
      Stream.changesWith((a, b) => a.label === b.label && a.canStop === b.canStop),
      Stream.runForEach(value => shell.presentModel(value).pipe(Effect.catchAll(Effect.logError))), Effect.forkScoped,
    )
  }
  const resolveQuitFailure = Atom.fn((decision: QuitFailureDecision) => Option.match(host.shell, {
    onNone: () => Effect.fail(unavailable),
    onSome: shell => shell.resolveQuitFailure(decision).pipe(Effect.ensuring(Effect.sync(() => registry.set(quitFailed, false)))),
  }))
  const retryService = Atom.fn((_: void) => Option.match(host.shell, { onNone: () => Effect.fail(unavailable), onSome: shell => shell.retryService }))


  const application = Atom.keepAlive(Atom.make(Option.match(host.shell, { onNone: () => Stream.fail(unavailable), onSome: shell => shell.application })))

  return {
    page: router.page,
    mainPage: router.mainPage,
    visit: router.visit,
    onboardingPending: onboardingPending as Atom.Atom<Option.Option<boolean>>,
    navigate,
    completeOnboarding,
    notices: notices as Atom.Atom<ReadonlyArray<HostNotice>>,
    dismissNotice,
    quitFailed: quitFailed as Atom.Atom<boolean>,
    addModelRequest: addModelRequest as Atom.Atom<Option.Option<AddModelRequest>>,
    requestAddModel,
    clearAddModelRequest,
    resolveQuitFailure,
    retryService,
    readAppearance: host.appearance.read,
    saveAppearance: (preference: AppearancePreference) => host.appearance.save(preference),
    clientWindow: host.window,
    openPathDialog: Option.map(host.shell, shell => shell.chooseOpenPath),
    application,
  }
})
export interface ApplicationSession extends Effect.Effect.Success<typeof makeApplicationSession> {}
export const ApplicationSession = Context.GenericTag<ApplicationSession>("client/ApplicationSession")
export const ApplicationSessionLive = Layer.scoped(ApplicationSession, makeApplicationSession)
