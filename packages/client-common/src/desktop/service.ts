import { OnboardingPersistence } from "../onboarding/persistence"
import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import type { LoginStartupState } from "@magnitudedev/sdk/desktop-host"
import type { LocalModelsState } from "@magnitudedev/sdk"
import type { DesktopConnectRequest, DesktopConnectionsSnapshot } from "./connections"
import type { HarnessId } from "../harness-connections/service"
import { LocalModels } from "../local-models/service"
import { LOCAL_MODEL_RANKING_SCALE_VALUES } from "../local-models/options"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import type { DesktopUpdateState } from "./update"

export const DesktopPage = Schema.Literal("discover", "catalog", "models", "connections", "status", "settings")
export type DesktopPage = typeof DesktopPage.Type
export const DesktopAction = Schema.Union(Schema.TaggedStruct("Navigate", { page: DesktopPage }), Schema.TaggedStruct("StopModel", {}))
export const SetupStatus = Schema.Literal("Required", "Complete", "Unavailable")
export const ModelTrayPresentation = Schema.Struct({ label: Schema.String, canStop: Schema.Boolean })
export const DesktopApplicationInfo = Schema.Struct({ version: Schema.String })
export class DesktopHostUnavailable extends Schema.TaggedError<DesktopHostUnavailable>()("DesktopHostUnavailable", {}) {
  override get message() { return "Desktop host unavailable" }
}
export interface DesktopBridge {
  readonly applicationInfo: Effect.Effect<typeof DesktopApplicationInfo.Type, unknown>
  readonly updates: Stream.Stream<DesktopUpdateState, unknown>
  readonly checkUpdate: Effect.Effect<void, unknown>
  readonly downloadUpdate: Effect.Effect<void, unknown>
  readonly restartUpdate: Effect.Effect<void, unknown>
  readonly loginStartup: Stream.Stream<LoginStartupState, unknown>
  readonly setLoginStartup: (enabled: boolean) => Effect.Effect<void, unknown>
  readonly connections: Stream.Stream<DesktopConnectionsSnapshot, unknown>
  readonly connect: (input: DesktopConnectRequest) => Effect.Effect<void, unknown>
  readonly disconnect: (harness: HarnessId) => Effect.Effect<void, unknown>
  readonly actions: Stream.Stream<typeof DesktopAction.Type>
  readonly presentSetup: (status: typeof SetupStatus.Type) => Effect.Effect<void, unknown>
  readonly presentModel: (value: typeof ModelTrayPresentation.Type) => Effect.Effect<void, unknown>
}
export const DesktopBridge = Context.GenericTag<Option.Option<DesktopBridge>>("client/DesktopBridge")
export const activeLocalModel = (models: LocalModelsState) => {
  for (const model of models.models) {
    const residency = model._tag === "Catalog" ? ("residencyState" in model.acquisitionState ? model.acquisitionState.residencyState : undefined) : model.state._tag === "Ready" ? model.state.residencyState : undefined
    if (residency && residency._tag !== "Unloaded" && residency._tag !== "Failed") {
      return Option.some({ model, residency })
    }
  }
  return Option.none()
}
export const modelTrayPresentation = (models: LocalModelsState): typeof ModelTrayPresentation.Type => {
  const active = activeLocalModel(models)
  if (Option.isSome(active)) {
    const { model, residency } = active.value
    return { label: `${formatLocalModelDisplayName(model)} · ${residency._tag === "Ready" ? "Loaded" : residency._tag === "Requested" ? "Loading" : residency._tag}`, canStop: true }
  }
  return { label: models.models.length === 0 && !models.preparation.assessment.complete ? "Reading model status…" : "No model loaded", canStop: false }
}
const makeDesktopSession = Effect.gen(function* () {
  const registry = yield* Registry.AtomRegistry
  const bridge = yield* DesktopBridge
  const page = Atom.keepAlive(Atom.make<DesktopPage>("discover"))
  const rankingPreference = Atom.keepAlive(Atom.make(2))
  const setRankingPreference = (index: number) => Effect.sync(() => {
    if (Number.isInteger(index) && index >= 0 && index < LOCAL_MODEL_RANKING_SCALE_VALUES.length) registry.set(rankingPreference, index)
  })
  const navigate = (value: DesktopPage) => Effect.sync(() => registry.set(page, value))
  if (Option.isSome(bridge)) {
    const host = bridge.value
    const onboarding = yield* OnboardingPersistence
    yield* Registry.toStream(registry, onboarding.state).pipe(
      Stream.map(result => Result.isSuccess(result) ? result.value.completed ? "Complete" as const : "Required" as const : "Unavailable" as const),
      Stream.changes, Stream.runForEach(status => host.presentSetup(status).pipe(Effect.catchAll(Effect.logError))), Effect.forkScoped,
    )
    const models = yield* LocalModels
    yield* host.actions.pipe(Stream.runForEach(action => action._tag === "Navigate" ? navigate(action.page) : models.stop.pipe(Effect.asVoid, Effect.catchAll(Effect.logError))), Effect.forkScoped)
    yield* Registry.toStream(registry, models.state).pipe(
      Stream.map(result => Result.isSuccess(result) ? modelTrayPresentation(result.value) : { label: "Model status unavailable", canStop: false }),
      Stream.changesWith((a, b) => a.label === b.label && a.canStop === b.canStop),
      Stream.runForEach(value => host.presentModel(value).pipe(Effect.catchAll(Effect.logError))), Effect.forkScoped,
    )
  }
  const loginStartup = Atom.make(Option.isSome(bridge) ? bridge.value.loginStartup : Stream.succeed({ _tag: "Unavailable" as const, message: "Desktop host unavailable" }))
  const applicationInfo = Atom.make(Option.isSome(bridge) ? bridge.value.applicationInfo : Effect.fail(new DesktopHostUnavailable()))
  const updates = Atom.make(Option.isSome(bridge) ? bridge.value.updates : Stream.succeed({ _tag: "Unavailable" as const, message: "Desktop host unavailable" }))
  const checkUpdate = Atom.fn(() => Option.isSome(bridge) ? bridge.value.checkUpdate : Effect.fail(new DesktopHostUnavailable()))
  const downloadUpdate = Atom.fn(() => Option.isSome(bridge) ? bridge.value.downloadUpdate : Effect.fail(new DesktopHostUnavailable()))
  const restartUpdate = Atom.fn(() => Option.isSome(bridge) ? bridge.value.restartUpdate : Effect.fail(new DesktopHostUnavailable()))
  const setLoginStartup = Atom.fn((enabled: boolean) => Option.isSome(bridge) ? bridge.value.setLoginStartup(enabled) : Effect.fail(new DesktopHostUnavailable()))
  const connections = Atom.make(Option.isSome(bridge) ? bridge.value.connections : Stream.succeed({ _tag: "Ready" as const, connections: [] }))
  const connect = Atom.fn((input: DesktopConnectRequest) => Option.isSome(bridge) ? bridge.value.connect(input) : Effect.fail(new DesktopHostUnavailable()))
  const disconnect = Atom.fn((harness: HarnessId) => Option.isSome(bridge) ? bridge.value.disconnect(harness) : Effect.fail(new DesktopHostUnavailable()))
  return { page: page as Atom.Atom<DesktopPage>, navigate, rankingPreference: rankingPreference as Atom.Atom<number>, setRankingPreference, applicationInfo, updates, checkUpdate, downloadUpdate, restartUpdate, loginStartup, setLoginStartup, connections, connect, disconnect }
})
export interface DesktopSession extends Effect.Effect.Success<typeof makeDesktopSession> {}
export const DesktopSession = Context.GenericTag<DesktopSession>("client/DesktopSession")
export const DesktopSessionLive = Layer.scoped(DesktopSession, makeDesktopSession)
