import { Rpc, RpcClient, RpcClientError, RpcGroup } from "@effect/rpc"
import { Schema } from "effect"
import {
  MenuActionSchema,
  type MenuAction,
} from "@magnitudedev/client-common/types/menu-action"
import {
  BrowserDownloadIdSchema,
  BrowserPermissionRequestIdSchema,
  BrowserTabIdSchema,
  BrowserViewportRectSchema,
  BrowserWorkspaceStateSchema,
  type BrowserDownloadId,
  type BrowserPermissionRequestId,
  type BrowserTabId,
  type BrowserViewportRect,
} from "@magnitudedev/client-common/platform/embedded-browser"
import {
  AcnEnsureEventSchema,
  AcnEnsureRequestSchema,
  AcnEnsuranceError,
  type AcnEnsureRequest,
} from "@magnitudedev/sdk"

export type { MenuAction }

export const DesktopRpcChannel = {
  request: "__magnitude:desktop-rpc:request",
  response: "__magnitude:desktop-rpc:response",
} as const

const Unit = Schema.Struct({})

export class DesktopRpcError extends Schema.TaggedError<DesktopRpcError>()(
  "DesktopRpcError",
  { message: Schema.String },
) {}

export const OpenFileOptionsPayload = Schema.Struct({
  multiple: Schema.optionalWith(Schema.Boolean, { default: () => false }),
})

export interface OpenFileOptions {
  readonly multiple?: boolean
}

export const DesktopRpcs = RpcGroup.make(
  Rpc.make("AcnEnsure", {
    payload: AcnEnsureRequestSchema,
    success: AcnEnsureEventSchema,
    error: AcnEnsuranceError,
    stream: true,
  }),
  Rpc.make("StorageGet", {
    payload: Schema.Struct({ key: Schema.String }),
    success: Schema.NullOr(Schema.String),
    error: DesktopRpcError,
  }),
  Rpc.make("StorageSet", {
    payload: Schema.Struct({ key: Schema.String, value: Schema.String }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("StorageRemove", {
    payload: Schema.Struct({ key: Schema.String }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("DialogOpenDirectory", {
    payload: Unit,
    success: Schema.NullOr(Schema.String),
    error: DesktopRpcError,
  }),
  Rpc.make("DialogOpenFile", {
    payload: OpenFileOptionsPayload,
    success: Schema.NullOr(Schema.Array(Schema.String)),
    error: DesktopRpcError,
  }),
  Rpc.make("NotificationShow", {
    payload: Schema.Struct({ title: Schema.String, body: Schema.String }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("Quit", {
    payload: Unit,
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("InterruptStream", {
    payload: Unit,
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("StreamMenuActions", {
    payload: Unit,
    success: MenuActionSchema,
    error: DesktopRpcError,
    stream: true,
  }),
  Rpc.make("BrowserObserve", {
    payload: Unit,
    success: BrowserWorkspaceStateSchema,
    error: DesktopRpcError,
    stream: true,
  }),
  Rpc.make("BrowserCreateTab", {
    payload: Schema.Struct({ url: Schema.NullOr(Schema.String) }),
    success: BrowserTabIdSchema,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserActivateTab", {
    payload: Schema.Struct({ tabId: BrowserTabIdSchema }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserCloseTab", {
    payload: Schema.Struct({ tabId: BrowserTabIdSchema }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserNavigate", {
    payload: Schema.Struct({ input: Schema.String }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserGoBack", { payload: Unit, success: Unit, error: DesktopRpcError }),
  Rpc.make("BrowserGoForward", { payload: Unit, success: Unit, error: DesktopRpcError }),
  Rpc.make("BrowserReload", { payload: Unit, success: Unit, error: DesktopRpcError }),
  Rpc.make("BrowserStop", { payload: Unit, success: Unit, error: DesktopRpcError }),
  Rpc.make("BrowserContinueInsecureNavigation", {
    payload: Unit,
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserCancelInsecureNavigation", {
    payload: Unit,
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserSetViewport", {
    payload: Schema.Struct({ bounds: Schema.NullOr(BrowserViewportRectSchema) }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserOpenExternal", { payload: Unit, success: Unit, error: DesktopRpcError }),
  Rpc.make("BrowserRespondToPermission", {
    payload: Schema.Struct({
      requestId: BrowserPermissionRequestIdSchema,
      allow: Schema.Boolean,
    }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserCancelDownload", {
    payload: Schema.Struct({ downloadId: BrowserDownloadIdSchema }),
    success: Unit,
    error: DesktopRpcError,
  }),
  Rpc.make("BrowserRevealDownload", {
    payload: Schema.Struct({ downloadId: BrowserDownloadIdSchema }),
    success: Unit,
    error: DesktopRpcError,
  }),
)

export type DesktopRpcClient = RpcClient.FromGroup<
  typeof DesktopRpcs,
  RpcClientError.RpcClientError
>

/**
 * Values exposed through contextBridge are structured-cloned, so Effect data
 * types (notably Option) must cross that boundary in their encoded form.
 */
export type DesktopAcnEnsureEvent = Schema.Schema.Encoded<
  typeof AcnEnsureEventSchema
>
export const encodeDesktopAcnEnsureEvent =
  Schema.encodeSync(AcnEnsureEventSchema)
export const decodeDesktopAcnEnsureEvent =
  Schema.decodeUnknownSync(AcnEnsureEventSchema)

export type DesktopBrowserWorkspaceState = Schema.Schema.Encoded<
  typeof BrowserWorkspaceStateSchema
>
export const encodeDesktopBrowserWorkspaceState =
  Schema.encodeSync(BrowserWorkspaceStateSchema)
export const decodeDesktopBrowserWorkspaceState =
  Schema.decodeUnknownSync(BrowserWorkspaceStateSchema)

export type DesktopPlatform = "darwin" | "win32" | "linux"

export interface DesktopApi {
  readonly platform: DesktopPlatform
  readonly acnEnsurer: {
    ensure(
      request: AcnEnsureRequest,
      onEvent: (event: DesktopAcnEnsureEvent) => void,
      onError: (error: unknown) => void,
      onEnd: () => void,
    ): () => void
  }
  readonly onMenuAction: (cb: (action: MenuAction) => void) => () => void
  readonly quit: () => void
  readonly interruptStream: () => void
  readonly openPath: (path: string) => Promise<void>
  readonly openExternal: (url: string) => Promise<void>
  readonly showItemInFolder?: (path: string) => void
  readonly storage: {
    getItem(key: string): Promise<string | null>
    setItem(key: string, value: string): Promise<void>
    removeItem(key: string): Promise<void>
  }
  readonly clipboard: {
    readText(): Promise<string>
    writeText(text: string): Promise<void>
  }
  readonly dialogs: {
    openDirectory(): Promise<string | null>
    openFile(options?: OpenFileOptions): Promise<string[] | null>
  }
  readonly notifications: {
    show(title: string, body: string): void
  }
  readonly browser: {
    observe(
      onState: (state: DesktopBrowserWorkspaceState) => void,
      onError: (error: unknown) => void,
      onEnd: () => void,
    ): () => void
    createTab(url?: string): Promise<BrowserTabId>
    activateTab(tabId: BrowserTabId): Promise<void>
    closeTab(tabId: BrowserTabId): Promise<void>
    navigate(input: string): Promise<void>
    goBack(): Promise<void>
    goForward(): Promise<void>
    reload(): Promise<void>
    stop(): Promise<void>
    continueInsecureNavigation(): Promise<void>
    cancelInsecureNavigation(): Promise<void>
    setViewport(bounds: BrowserViewportRect | null): Promise<void>
    openExternal(): Promise<void>
    respondToPermission(
      requestId: BrowserPermissionRequestId,
      allow: boolean,
    ): Promise<void>
    cancelDownload(downloadId: BrowserDownloadId): Promise<void>
    revealDownload(downloadId: BrowserDownloadId): Promise<void>
  }
}
