import { Schema } from "effect"

export const BrowserTabIdSchema = Schema.NonEmptyString.pipe(
  Schema.brand("BrowserTabId"),
)
export type BrowserTabId = typeof BrowserTabIdSchema.Type

export const BrowserPermissionRequestIdSchema = Schema.NonEmptyString.pipe(
  Schema.brand("BrowserPermissionRequestId"),
)
export type BrowserPermissionRequestId =
  typeof BrowserPermissionRequestIdSchema.Type

export const BrowserDownloadIdSchema = Schema.NonEmptyString.pipe(
  Schema.brand("BrowserDownloadId"),
)
export type BrowserDownloadId = typeof BrowserDownloadIdSchema.Type

export const BrowserTabPhaseSchema = Schema.Literal(
  "blank",
  "loading",
  "ready",
  "failed",
  "crashed",
  "discarded",
  "insecure",
)
export type BrowserTabPhase = typeof BrowserTabPhaseSchema.Type

export const BrowserTabStateSchema = Schema.Struct({
  id: BrowserTabIdSchema,
  title: Schema.String,
  url: Schema.String,
  pendingUrl: Schema.NullOr(Schema.String),
  faviconUrl: Schema.NullOr(Schema.String),
  phase: BrowserTabPhaseSchema,
  canGoBack: Schema.Boolean,
  canGoForward: Schema.Boolean,
  error: Schema.NullOr(Schema.String),
  insecureUrl: Schema.NullOr(Schema.String),
})
export type BrowserTabState = typeof BrowserTabStateSchema.Type

export const BrowserPermissionRequestSchema = Schema.Struct({
  id: BrowserPermissionRequestIdSchema,
  tabId: BrowserTabIdSchema,
  origin: Schema.String,
  permission: Schema.String,
})
export type BrowserPermissionRequest =
  typeof BrowserPermissionRequestSchema.Type

export const BrowserDownloadStateSchema = Schema.Struct({
  id: BrowserDownloadIdSchema,
  tabId: BrowserTabIdSchema,
  fileName: Schema.String,
  savePath: Schema.NullOr(Schema.String),
  receivedBytes: Schema.Number,
  totalBytes: Schema.Number,
  status: Schema.Literal("progressing", "completed", "cancelled", "failed"),
})
export type BrowserDownloadState = typeof BrowserDownloadStateSchema.Type

export const BrowserWorkspaceStateSchema = Schema.Struct({
  revision: Schema.Int,
  focusLocationRevision: Schema.Int,
  activeTabId: BrowserTabIdSchema,
  tabs: Schema.Array(BrowserTabStateSchema),
  permissionRequest: Schema.NullOr(BrowserPermissionRequestSchema),
  downloads: Schema.Array(BrowserDownloadStateSchema),
})
export type BrowserWorkspaceState = typeof BrowserWorkspaceStateSchema.Type

export const BrowserViewportRectSchema = Schema.Struct({
  x: Schema.Int,
  y: Schema.Int,
  width: Schema.Int.pipe(Schema.nonNegative()),
  height: Schema.Int.pipe(Schema.nonNegative()),
})
export type BrowserViewportRect = typeof BrowserViewportRectSchema.Type

/**
 * Desktop-hosted browser capability. Runtime truth is owned by Electron's main
 * process; this interface exposes only an observed snapshot and semantic
 * commands to the shared React client.
 */
export interface EmbeddedBrowserCapability {
  readonly getSnapshot: () => BrowserWorkspaceState
  readonly subscribe: (listener: () => void) => () => void
  readonly createTab: (url?: string) => Promise<void>
  readonly activateTab: (tabId: BrowserTabId) => Promise<void>
  readonly closeTab: (tabId: BrowserTabId) => Promise<void>
  readonly navigate: (input: string) => Promise<void>
  readonly goBack: () => Promise<void>
  readonly goForward: () => Promise<void>
  readonly reload: () => Promise<void>
  readonly stop: () => Promise<void>
  readonly continueInsecureNavigation: () => Promise<void>
  readonly cancelInsecureNavigation: () => Promise<void>
  readonly setViewport: (bounds: BrowserViewportRect | null) => Promise<void>
  readonly openExternal: () => Promise<void>
  readonly respondToPermission: (
    requestId: BrowserPermissionRequestId,
    allow: boolean,
  ) => Promise<void>
  readonly cancelDownload: (downloadId: BrowserDownloadId) => Promise<void>
  readonly revealDownload: (downloadId: BrowserDownloadId) => Promise<void>
}
