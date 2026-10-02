import { Rpc, RpcGroup, type RpcClient, type RpcClientError } from "@effect/rpc"
import { atMostOnce, replaySafe } from "@magnitudedev/sdk"
import { AppearancePreference, ApplicationSnapshot, LoginStartupState, DesktopUpdateState } from "@magnitudedev/sdk/desktop-host"
import { ModelTrayPresentation, HostAction as ApplicationAction, QuitFailureDecision } from "@magnitudedev/client-common/application/contracts"
import { Schema } from "effect"

export { ApplicationPage as Page, ModelTrayPresentation, HostAction as ApplicationAction, HostNotice } from "@magnitudedev/client-common/application/contracts"
export class HostError extends Schema.TaggedError<HostError>()("HostError", { message: Schema.String }) {}
const Unit = Schema.Struct({})
export const InferenceHostRpcs = RpcGroup.make(
  Rpc.make("Updates", { payload: Unit, success: DesktopUpdateState, error: HostError, stream: true }),
  Rpc.make("SetAutoDownload", { payload: Schema.Struct({ enabled: Schema.Boolean }), success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("CheckUpdate", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("DiscardUpdate", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("DownloadUpdate", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("RestartUpdate", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Observe", { payload: Unit, success: ApplicationSnapshot, error: HostError, stream: true }),
  Rpc.make("Actions", { payload: Unit, success: ApplicationAction, error: HostError, stream: true }),
  Rpc.make("PresentModel", { payload: ModelTrayPresentation, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("GetAppearance", { payload: Unit, success: AppearancePreference, error: HostError }).pipe(replaySafe),
  Rpc.make("SetAppearance", { payload: Schema.Struct({ preference: AppearancePreference }), success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Relaunch", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("LoginStartup", { payload: Unit, success: LoginStartupState, error: HostError, stream: true }),
  Rpc.make("SetLoginStartup", { payload: Schema.Struct({ enabled: Schema.Boolean }), success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Retry", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Quit", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("ResolveQuitFailure", { payload: Schema.Struct({ decision: QuitFailureDecision }), success: Unit, error: HostError }).pipe(atMostOnce),
)
export type InferenceHostClient = RpcClient.FromGroup<typeof InferenceHostRpcs, RpcClientError.RpcClientError>
export interface DesktopApi {
  readonly updates: (value: (state: typeof DesktopUpdateState.Type) => void, error: (message: string) => void) => () => void
  readonly setAutoDownload: (enabled: boolean) => Promise<void>
  readonly checkUpdate: () => Promise<void>
  readonly discardUpdate: () => Promise<void>
  readonly downloadUpdate: () => Promise<void>
  readonly restartUpdate: () => Promise<void>
  readonly platform: string
  readonly observe: (value: (snapshot: typeof ApplicationSnapshot.Encoded) => void, error: (message: string) => void) => () => void
  readonly actions: (value: (action: typeof ApplicationAction.Type) => void) => () => void
  readonly presentModel: (value: typeof ModelTrayPresentation.Encoded) => Promise<void>
  readonly getAppearance: () => Promise<AppearancePreference>
  readonly setAppearance: (preference: AppearancePreference) => Promise<void>
  readonly relaunch: () => Promise<void>
  readonly loginStartup: (value: (state: typeof LoginStartupState.Type) => void, error: (message: string) => void) => () => void
  readonly setLoginStartup: (enabled: boolean) => Promise<void>
  readonly retry: () => Promise<void>
  readonly quit: () => Promise<void>
  readonly resolveQuitFailure: (decision: QuitFailureDecision) => Promise<void>
}

export const DesktopRpcChannel = { request: "__magnitude:desktop-rpc:request", response: "__magnitude:desktop-rpc:response" } as const
export const DesktopRendererSession = Schema.UUID.pipe(Schema.brand("DesktopRendererSession"))
/** The nested message remains owned by Effect RPC's protocol and operation schemas. */
export const DesktopRpcEnvelope = Schema.Struct({ session: DesktopRendererSession, message: Schema.Unknown })
