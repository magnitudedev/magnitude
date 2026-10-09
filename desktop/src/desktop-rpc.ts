import { Rpc, RpcGroup, type RpcClient, type RpcClientError } from "@effect/rpc"
import { atMostOnce, replaySafe } from "@magnitudedev/sdk"
import { AppearancePreference, ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"
import { ModelTrayPresentation, HostAction as ApplicationAction, QuitFailureDecision } from "@magnitudedev/client-common/application/contracts"
import { Schema } from "effect"

export { ApplicationPage as Page, ModelTrayPresentation, HostAction as ApplicationAction, HostNotice } from "@magnitudedev/client-common/application/contracts"
export class HostError extends Schema.TaggedError<HostError>()("HostError", { message: Schema.String }) {}
const Unit = Schema.Struct({})
export const InferenceHostRpcs = RpcGroup.make(
  Rpc.make("Observe", { payload: Unit, success: ApplicationSnapshot, error: HostError, stream: true }),
  Rpc.make("Actions", { payload: Unit, success: ApplicationAction, error: HostError, stream: true }),
  Rpc.make("PresentModel", { payload: ModelTrayPresentation, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("GetAppearance", { payload: Unit, success: AppearancePreference, error: HostError }).pipe(replaySafe),
  Rpc.make("SetAppearance", { payload: Schema.Struct({ preference: AppearancePreference }), success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("Retry", { payload: Unit, success: Unit, error: HostError }).pipe(atMostOnce),
  Rpc.make("ResolveQuitFailure", { payload: Schema.Struct({ decision: QuitFailureDecision }), success: Unit, error: HostError }).pipe(atMostOnce),
)
export type InferenceHostClient = RpcClient.FromGroup<typeof InferenceHostRpcs, RpcClientError.RpcClientError>
export interface DesktopApi {
  readonly platform: string
  readonly observe: (value: (snapshot: typeof ApplicationSnapshot.Encoded) => void, error: (message: string) => void) => () => void
  readonly actions: (value: (action: typeof ApplicationAction.Type) => void) => () => void
  readonly presentModel: (value: typeof ModelTrayPresentation.Encoded) => Promise<void>
  readonly getAppearance: () => Promise<AppearancePreference>
  readonly setAppearance: (preference: AppearancePreference) => Promise<void>
  readonly retry: () => Promise<void>
  readonly resolveQuitFailure: (decision: QuitFailureDecision) => Promise<void>
}

export const DesktopRpcChannel = { request: "__magnitude:desktop-rpc:request", response: "__magnitude:desktop-rpc:response" } as const
export const DesktopRendererSession = Schema.UUID.pipe(Schema.brand("DesktopRendererSession"))
/** The nested message remains owned by Effect RPC's protocol and operation schemas. */
export const DesktopRpcEnvelope = Schema.Struct({ session: DesktopRendererSession, message: Schema.Unknown })
