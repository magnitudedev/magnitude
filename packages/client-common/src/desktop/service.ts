import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import type { LoginStartupState, ApplicationMemoryObservation, MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import { ModelLoadStageSchema, type LocalModelsState, type ModelResidency } from "@magnitudedev/sdk"
import type { DesktopConnectRequest, DesktopConnectionsSnapshot } from "./connections"
import type { HarnessId } from "../harness-connections/service"
import { LocalModels } from "../local-models/service"
import { LOCAL_MODEL_RANKING_SCALE_VALUES } from "../local-models/options"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import { formatModelLoadPercentage, formatModelLoadStage, formatModelMemory, isMeasuredModelLoadStage } from "../utils/model-load"
import type { DesktopUpdateState } from "./update"
import { DesktopPage, DesktopAction, ModelTrayPresentation, ModelTrayStatus, DesktopApplicationInfo } from "./contracts"

export { DesktopPage, DesktopAction, ModelTrayPresentation, ModelTrayStatus, DesktopApplicationInfo } from "./contracts"
export class DesktopHostUnavailable extends Schema.TaggedError<DesktopHostUnavailable>()("DesktopHostUnavailable", {}) {
  override get message() { return "Desktop host unavailable" }
}
export interface DesktopBridge {
  readonly machineIdentity: Effect.Effect<MachineIdentityObservation, unknown>
  readonly memory: Stream.Stream<ApplicationMemoryObservation, unknown>
  readonly applicationInfo: Effect.Effect<typeof DesktopApplicationInfo.Type, unknown>
  readonly updates: Stream.Stream<DesktopUpdateState, unknown>
  readonly setAutoDownload: (enabled: boolean) => Effect.Effect<void, unknown>
  readonly checkUpdate: Effect.Effect<void, unknown>
  readonly discardUpdate: Effect.Effect<void, unknown>
  readonly downloadUpdate: Effect.Effect<void, unknown>
  readonly restartUpdate: Effect.Effect<void, unknown>
  readonly loginStartup: Stream.Stream<LoginStartupState, unknown>
  readonly setLoginStartup: (enabled: boolean) => Effect.Effect<void, unknown>
  readonly connections: Stream.Stream<DesktopConnectionsSnapshot, unknown>
  readonly connect: (input: DesktopConnectRequest) => Effect.Effect<void, unknown>
  readonly disconnect: (harness: HarnessId) => Effect.Effect<void, unknown>
  readonly actions: Stream.Stream<typeof DesktopAction.Type>
  readonly presentModel: (value: typeof ModelTrayPresentation.Type) => Effect.Effect<void, unknown>
}
export const DesktopBridge = Context.GenericTag<Option.Option<DesktopBridge>>("client/DesktopBridge")
export const activeLocalModel = (models: LocalModelsState) => {
  for (const model of models.models) {
    const residency = model._tag === "Catalog" ? ("residencyState" in model.acquisitionState ? model.acquisitionState.residencyState : undefined) : model.state._tag === "Ready" ? model.state.residencyState : undefined
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
    const models = yield* LocalModels
    yield* host.actions.pipe(Stream.runForEach(action => action._tag === "Navigate" ? navigate(action.page) : models.stop.pipe(Effect.asVoid, Effect.catchAll(Effect.logError))), Effect.forkScoped)
    yield* Registry.toStream(registry, models.state).pipe(
      Stream.map(result => Result.isSuccess(result) ? modelTrayPresentation(result.value) : { label: "Model status unavailable", status: Option.none(), canStop: false }),
      Stream.changesWith((a, b) => a.label === b.label && a.canStop === b.canStop),
      Stream.runForEach(value => host.presentModel(value).pipe(Effect.catchAll(Effect.logError))), Effect.forkScoped,
    )
  }
  const loginStartup = Atom.make(Option.isSome(bridge) ? bridge.value.loginStartup : Stream.succeed({ _tag: "Unavailable" as const, message: "Desktop host unavailable" }))
  const machineIdentity = Atom.keepAlive(Atom.make(Option.isSome(bridge) ? bridge.value.machineIdentity : Effect.succeed({ _tag: "Unavailable" as const, formFactor: "Unknown" as const })))
  const memory = Atom.make(Option.isSome(bridge) ? bridge.value.memory : Stream.succeed({ _tag: "Unavailable" as const, message: "Desktop host unavailable" }))
  const applicationInfo = Atom.make(Option.isSome(bridge) ? bridge.value.applicationInfo : Effect.fail(new DesktopHostUnavailable()))
  const updates = Atom.make(Option.isSome(bridge) ? bridge.value.updates : Stream.succeed({ transfer: { _tag: "Unavailable" as const, message: "Desktop host unavailable" }, check: { _tag: "Idle" as const }, preference: { _tag: "Unavailable" as const, message: "Desktop host unavailable" } }))
  const setAutoDownload = Atom.fn((enabled: boolean) => Option.isSome(bridge) ? bridge.value.setAutoDownload(enabled) : Effect.fail(new DesktopHostUnavailable()))
  const checkUpdate = Atom.fn(() => Option.isSome(bridge) ? bridge.value.checkUpdate : Effect.fail(new DesktopHostUnavailable()))
  const discardUpdate = Atom.fn(() => Option.isSome(bridge) ? bridge.value.discardUpdate : Effect.fail(new DesktopHostUnavailable()))
  const downloadUpdate = Atom.fn(() => Option.isSome(bridge) ? bridge.value.downloadUpdate : Effect.fail(new DesktopHostUnavailable()))
  const restartUpdate = Atom.fn(() => Option.isSome(bridge) ? bridge.value.restartUpdate : Effect.fail(new DesktopHostUnavailable()))
  const setLoginStartup = Atom.fn((enabled: boolean) => Option.isSome(bridge) ? bridge.value.setLoginStartup(enabled) : Effect.fail(new DesktopHostUnavailable()))
  const connections = Atom.make(Option.isSome(bridge) ? bridge.value.connections : Stream.succeed({ _tag: "Ready" as const, connections: [] }))
  const connect = Atom.fn((input: DesktopConnectRequest) => Option.isSome(bridge) ? bridge.value.connect(input) : Effect.fail(new DesktopHostUnavailable()))
  const disconnect = Atom.fn((harness: HarnessId) => Option.isSome(bridge) ? bridge.value.disconnect(harness) : Effect.fail(new DesktopHostUnavailable()))
  return { machineIdentity, memory, page: page as Atom.Atom<DesktopPage>, navigate, rankingPreference: rankingPreference as Atom.Atom<number>, setRankingPreference, applicationInfo, updates, setAutoDownload, checkUpdate, downloadUpdate, discardUpdate, restartUpdate, loginStartup, setLoginStartup, connections, connect, disconnect }
})
export interface DesktopSession extends Effect.Effect.Success<typeof makeDesktopSession> {}
export const DesktopSession = Context.GenericTag<DesktopSession>("client/DesktopSession")
export const DesktopSessionLive = Layer.scoped(DesktopSession, makeDesktopSession)
