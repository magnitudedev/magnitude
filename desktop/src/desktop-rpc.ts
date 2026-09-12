import { Rpc, RpcGroup, type RpcClient, type RpcClientError } from "@effect/rpc"
import { atMostOnce, replaySafe } from "@magnitudedev/sdk"
import { ApplicationSnapshot, LoginStartupState, ApplicationMemoryObservation } from "@magnitudedev/sdk/desktop-host"
import { DesktopApplicationInfo, DesktopConnectRequest, DesktopConnectionsSnapshot, DesktopUpdateState, HarnessIdSchema } from "@magnitudedev/client-common"
import { Schema } from "effect"

export { DesktopPage as Page, ModelTrayPresentation, DesktopAction as ApplicationAction } from "@magnitudedev/client-common"
import { DesktopPage as Page, ModelTrayPresentation, DesktopAction as ApplicationAction } from "@magnitudedev/client-common"
export class HostError extends Schema.TaggedError<HostError>()("HostError", { message: Schema.String }) {}
const Unit = Schema.Struct({})
export const InferenceHostRpcs = RpcGroup.make(
  Rpc.make("ApplicationInfo", { payload: Unit, success: DesktopApplicationInfo, error: HostError }).pipe(replaySafe),
  Rpc.make("Memory", { payload: Unit, success: ApplicationMemoryObservation, error: HostError, stream: true }),
  Rpc.make("Updates", { payload: Unit, success: DesktopUpdateState, error: HostError, stream: true }),
  Rpc.make("CheckUpdate", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("DownloadUpdate", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("RestartUpdate", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Observe", { payload: Unit, success: ApplicationSnapshot, error: HostError, stream: true }),
  Rpc.make("Actions", { payload: Unit, success: ApplicationAction, error: HostError, stream: true }),
  Rpc.make("PresentModel", { payload: ModelTrayPresentation, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Appearance", { payload: Schema.Struct({ preference: Schema.Literal("system", "light", "dark") }), success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("LoginStartup", { payload: Unit, success: LoginStartupState, error: HostError, stream: true }),
  Rpc.make("SetLoginStartup", { payload: Schema.Struct({ enabled: Schema.Boolean }), success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Connections", { payload: Unit, success: DesktopConnectionsSnapshot, error: HostError, stream: true }),
  Rpc.make("Connect", { payload: DesktopConnectRequest, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Disconnect", { payload: Schema.Struct({ harness: HarnessIdSchema }), success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Retry", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Quit", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
)
export type InferenceHostClient = RpcClient.FromGroup<typeof InferenceHostRpcs, RpcClientError.RpcClientError>
export interface DesktopApi {
  readonly memory: (value: (state: ApplicationMemoryObservation) => void, error: (message: string) => void) => () => void
  readonly applicationInfo: () => Promise<typeof DesktopApplicationInfo.Type>
  readonly updates: (value: (state: typeof DesktopUpdateState.Type) => void, error: (message: string) => void) => () => void
  readonly checkUpdate: () => Promise<void>
  readonly downloadUpdate: () => Promise<void>
  readonly restartUpdate: () => Promise<void>
  readonly platform: string
  readonly observe: (value: (snapshot: typeof ApplicationSnapshot.Encoded) => void, error: (message: string) => void) => () => void
  readonly actions: (value: (action: typeof ApplicationAction.Type) => void) => () => void
  readonly presentModel: (value: typeof ModelTrayPresentation.Type) => Promise<void>
  readonly appearance: (preference: "system" | "light" | "dark") => Promise<void>
  readonly loginStartup: (value: (state: typeof LoginStartupState.Type) => void, error: (message: string) => void) => () => void
  readonly setLoginStartup: (enabled: boolean) => Promise<void>
  readonly connections: (value: (rows: typeof DesktopConnectionsSnapshot.Encoded) => void, error: (message: string) => void) => () => void
  readonly connect: (input: typeof DesktopConnectRequest.Encoded) => Promise<void>
  readonly disconnect: (harness: typeof HarnessIdSchema.Type) => Promise<void>
  readonly retry: () => Promise<void>
  readonly quit: () => Promise<void>
}

export const DesktopRpcChannel = { request: "__magnitude:desktop-rpc:request", response: "__magnitude:desktop-rpc:response" } as const
export const DesktopRendererSession = Schema.UUID.pipe(Schema.brand("DesktopRendererSession"))
/** The nested message remains owned by Effect RPC's protocol and operation schemas. */
export const DesktopRpcEnvelope = Schema.Struct({ session: DesktopRendererSession, message: Schema.Unknown })
