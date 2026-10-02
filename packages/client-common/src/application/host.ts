import { Context, Option, Schema, type Effect, type Stream } from "effect"
import type {
  AppearancePreference,
  ApplicationMemoryObservation,
  ApplicationSnapshot,
  LoginStartupState,
  MachineIdentityObservation,
  ModelStorageSettings,
  NetworkAccessChange,
  NetworkAccessSettings,
} from "@magnitudedev/sdk/desktop-host"
import type { HarnessId } from "../harness-connections/service"
import type { DesktopConnectRequest, DesktopConnectionsSnapshot } from "./connections"
import type { DesktopUpdateState } from "./update"
import type { ApplicationInfo, HostAction, ModelTrayPresentation, QuitFailureDecision } from "./contracts"

export class ApplicationHostFailed extends Schema.TaggedError<ApplicationHostFailed>()("ApplicationHostFailed", {
  message: Schema.String,
}) {}

/** Window chrome of a native application window. */
export interface HostWindow {
  readonly platform: string
}

/** Where the viewer's appearance preference is kept: Electron main, or the browser. */
export interface HostAppearance {
  readonly read: Effect.Effect<AppearancePreference, ApplicationHostFailed>
  readonly save: (preference: AppearancePreference) => Effect.Effect<void, ApplicationHostFailed>
}

/** The native shell around the window: tray, menus, and requests raised by Electron main. */
export interface HostShell {
  readonly actions: Stream.Stream<HostAction, ApplicationHostFailed>
  readonly presentModel: (value: typeof ModelTrayPresentation.Type) => Effect.Effect<void, ApplicationHostFailed>
  readonly resolveQuitFailure: (decision: QuitFailureDecision) => Effect.Effect<void, ApplicationHostFailed>
  readonly retryService: Effect.Effect<void, ApplicationHostFailed>
}

/** Server and owner capabilities that Electron main still serves for the machine it runs on. */
export interface DesktopControls {
  readonly platform: string
  readonly application: Stream.Stream<ApplicationSnapshot, ApplicationHostFailed>
  readonly applicationInfo: Effect.Effect<typeof ApplicationInfo.Type, ApplicationHostFailed>
  readonly machineIdentity: Effect.Effect<MachineIdentityObservation, ApplicationHostFailed>
  readonly memory: Stream.Stream<ApplicationMemoryObservation, ApplicationHostFailed>
  readonly updates: Stream.Stream<DesktopUpdateState, ApplicationHostFailed>
  readonly setAutoDownload: (enabled: boolean) => Effect.Effect<void, ApplicationHostFailed>
  readonly checkUpdate: Effect.Effect<void, ApplicationHostFailed>
  readonly discardUpdate: Effect.Effect<void, ApplicationHostFailed>
  readonly downloadUpdate: Effect.Effect<void, ApplicationHostFailed>
  readonly restartUpdate: Effect.Effect<void, ApplicationHostFailed>
  readonly loginStartup: Stream.Stream<LoginStartupState, ApplicationHostFailed>
  readonly setLoginStartup: (enabled: boolean) => Effect.Effect<void, ApplicationHostFailed>
  readonly connections: Stream.Stream<DesktopConnectionsSnapshot, ApplicationHostFailed>
  readonly connect: (input: DesktopConnectRequest) => Effect.Effect<void, ApplicationHostFailed>
  readonly disconnect: (harness: HarnessId) => Effect.Effect<void, ApplicationHostFailed>
  readonly modelStorage: Effect.Effect<ModelStorageSettings, ApplicationHostFailed>
  readonly setModelStorage: (path: Option.Option<string>) => Effect.Effect<void, ApplicationHostFailed>
  readonly chooseModelStorageDirectory: Effect.Effect<Option.Option<string>, ApplicationHostFailed>
  readonly networkAccess: Effect.Effect<NetworkAccessSettings, ApplicationHostFailed>
  readonly setNetworkAccess: (change: NetworkAccessChange) => Effect.Effect<void, ApplicationHostFailed>
  readonly regenerateNetworkApiKey: Effect.Effect<void, ApplicationHostFailed>
  readonly relaunch: Effect.Effect<void, ApplicationHostFailed>
}

/**
 * What the machine showing the UI provides. Electron provides every group; a browser provides only
 * appearance. Pages show a control when its capability is present.
 */
export interface ApplicationHost {
  readonly window: Option.Option<HostWindow>
  readonly appearance: HostAppearance
  readonly shell: Option.Option<HostShell>
  readonly desktop: Option.Option<DesktopControls>
}
export const ApplicationHost = Context.GenericTag<ApplicationHost>("client/ApplicationHost")
