import { Context, Schema, type Effect, type Option, type Stream } from "effect"
import type { AppearancePreference, ApplicationSnapshot, LoginStartupState } from "@magnitudedev/sdk/desktop-host"
import type { DesktopUpdateState } from "./update"
import type { HostAction, ModelTrayPresentation, QuitFailureDecision } from "./contracts"

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

/** Owner capabilities that Electron main serves for the application it owns. */
export interface DesktopControls {
  readonly application: Stream.Stream<ApplicationSnapshot, ApplicationHostFailed>
  readonly updates: Stream.Stream<DesktopUpdateState, ApplicationHostFailed>
  readonly setAutoDownload: (enabled: boolean) => Effect.Effect<void, ApplicationHostFailed>
  readonly checkUpdate: Effect.Effect<void, ApplicationHostFailed>
  readonly discardUpdate: Effect.Effect<void, ApplicationHostFailed>
  readonly downloadUpdate: Effect.Effect<void, ApplicationHostFailed>
  readonly restartUpdate: Effect.Effect<void, ApplicationHostFailed>
  readonly loginStartup: Stream.Stream<LoginStartupState, ApplicationHostFailed>
  readonly setLoginStartup: (enabled: boolean) => Effect.Effect<void, ApplicationHostFailed>
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
