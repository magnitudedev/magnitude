import { Rpc, RpcClient, RpcClientError, RpcGroup } from "@effect/rpc"
import { Schema } from "effect"
import { MenuActionSchema, type MenuAction } from "@magnitudedev/client-common/src/types/menu-action"

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
  Rpc.make("DaemonEnsure", {
    payload: Unit,
    success: Schema.String,
    error: DesktopRpcError,
  }),
  Rpc.make("DaemonDiscover", {
    payload: Unit,
    success: Schema.NullOr(Schema.String),
    error: DesktopRpcError,
  }),
  Rpc.make("DaemonSpawn", {
    payload: Schema.Struct({ command: Schema.NullOr(Schema.Array(Schema.String)) }),
    success: Schema.String,
    error: DesktopRpcError,
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
)

export type DesktopRpcClient = RpcClient.FromGroup<typeof DesktopRpcs, RpcClientError.RpcClientError>

export type DesktopPlatform = "darwin" | "win32" | "linux"

export interface DesktopApi {
  /** Promise that resolves once main has ensured the daemon is reachable. */
  readonly ready: Promise<void>
  /** Promise that resolves with an error message if daemon connection fails. */
  readonly daemonError: Promise<string>
  readonly platform: DesktopPlatform
  readonly daemon: {
    discover(): Promise<string | null>
    spawn(command?: string[]): Promise<string>
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
}
