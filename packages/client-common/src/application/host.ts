import { Context, Schema, type Effect, type Option, type Stream } from "effect"
import type { AppearancePreference, ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"
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
  /** The desktop owner's own view of the service and tray, available even when ACN is down. */
  readonly application: Stream.Stream<ApplicationSnapshot, ApplicationHostFailed>
}

/**
 * What the machine showing the UI provides. Electron provides every group; a browser provides only
 * appearance. Pages show a control when its capability is present. Everything about the machine
 * running Magnitude, including its owner, comes from ACN instead.
 */
export interface ApplicationHost {
  readonly window: Option.Option<HostWindow>
  readonly appearance: HostAppearance
  readonly shell: Option.Option<HostShell>
}
export const ApplicationHost = Context.GenericTag<ApplicationHost>("client/ApplicationHost")
