import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { CatalogFormModelIdSchema, localModelResidency, ModelLoadStageSchema, type LocalModelsState, type ModelResidency } from "@magnitudedev/sdk"
import type { AppearancePreference } from "@magnitudedev/sdk/desktop-host"
import { LocalModels } from "../local-models/service"
import { LOCAL_MODEL_RANKING_SCALE_VALUES } from "../local-models/options"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import { formatModelLoadPercentage, formatModelLoadStage, formatModelMemory, isMeasuredModelLoadStage } from "../utils/model-load"
import { ApplicationHost, ApplicationHostFailed } from "./host"
import { ApplicationRouter } from "./router"
import type { AddModelRequest, ApplicationPage, HostNotice, QuitFailureDecision } from "./contracts"
import { ModelTrayPresentation, ModelTrayStatus } from "./contracts"

export { AddModelRequest, ApplicationPage, HostAction, HostNotice, QuitFailureDecision, ModelTrayPresentation, ModelTrayStatus } from "./contracts"

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

/** Connection-lifetime presentation state for Discover, independent of live model observations. */
export const DiscoverState = Schema.Struct({
  view: Schema.Literal("recommended", "all"),
  search: Schema.String,
  filter: Schema.Literal("all", "fits"),
  lab: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  sort: Schema.Literal("recommended", "name", "smallest", "largest"),
  machineExpanded: Schema.Boolean,
  selectedModel: Schema.optionalWith(CatalogFormModelIdSchema, { as: "Option", exact: true }),
  recommendedScroll: Schema.Number,
  allScroll: Schema.Number,
})
export type DiscoverState = typeof DiscoverState.Type

const unavailable = new ApplicationHostFailed({ message: "This action isn't available here." })

const makeApplicationSession = Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const host = yield* ApplicationHost
  const router = yield* ApplicationRouter
  const models = yield* LocalModels
  const noticeDismissed = yield* host.foundModelsNotice.readDismissed.pipe(Effect.catchAll(error => Effect.logWarning(error.message).pipe(Effect.as(false))))
  const foundModelsNoticeDismissed = Atom.keepAlive(Atom.make(noticeDismissed))
  const dismissFoundModelsNotice = Atom.fn((_: void) => host.foundModelsNotice.dismiss.pipe(
    Effect.tap(() => Effect.sync(() => registry.set(foundModelsNoticeDismissed, true))),
  ))
  const discover = Atom.keepAlive(Atom.make<DiscoverState>({
    view: registry.get(router.page) === "catalog" ? "all" : "recommended", search: "", filter: "all", lab: Option.none(), sort: "recommended",
    machineExpanded: false, selectedModel: Option.none(), recommendedScroll: 0, allScroll: 0,
  }))
  const updateDiscover = (changes: Partial<DiscoverState>) => Effect.sync(() => registry.set(discover, { ...registry.get(discover), ...changes }))
  // Catalog links and native menu actions open the All models view of Discover.
  const navigate = (page: ApplicationPage) => page === "catalog"
    ? updateDiscover({ view: "all" }).pipe(Effect.zipRight(router.navigate("discover")))
    : router.navigate(page)
  const rankingPreference = Atom.keepAlive(Atom.make(2))
  const setRankingPreference = (index: number) => Effect.sync(() => {
    if (Number.isInteger(index) && index >= 0 && index < LOCAL_MODEL_RANKING_SCALE_VALUES.length) registry.set(rankingPreference, index)
  })
  const notices = Atom.keepAlive(Atom.make<ReadonlyArray<HostNotice>>([]))
  const dismissNotice = (notice: HostNotice) => Effect.sync(() => registry.set(notices, registry.get(notices).filter(entry => entry !== notice)))
  const quitFailed = Atom.keepAlive(Atom.make(false))
  const addModelRequest = Atom.keepAlive(Atom.make(Option.none<AddModelRequest>()))
  const requestAddModel = (request: AddModelRequest) => navigate("discover").pipe(
    Effect.zipRight(Effect.sync(() => registry.set(addModelRequest, Option.some(request)))))
  const clearAddModelRequest = Effect.sync(() => registry.set(addModelRequest, Option.none()))

  if (Option.isSome(host.shell)) {
    const shell = host.shell.value
    yield* shell.actions.pipe(
      Stream.runForEach(action => {
        switch (action._tag) {
          case "Navigate": return navigate(action.page)
          case "StopModel": return models.stop.pipe(Effect.asVoid, Effect.catchAll(Effect.logError))
          case "ShowNotice": return Effect.sync(() => registry.set(notices, [...registry.get(notices), action.notice]))
          case "QuitFailed": return Effect.sync(() => registry.set(quitFailed, true))
          case "OpenHuggingFaceModel": return requestAddModel({ _tag: "HuggingFace", repository: action.repository, file: action.file })
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
    navigate,
    discover: discover as Atom.Atom<DiscoverState>,
    updateDiscover,
    foundModelsNoticeDismissed: foundModelsNoticeDismissed as Atom.Atom<boolean>,
    dismissFoundModelsNotice,
    rankingPreference: rankingPreference as Atom.Atom<number>,
    setRankingPreference,
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
