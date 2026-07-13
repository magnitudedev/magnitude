/**
 * Web package barrel exports — spec §4.3
 */
export { App } from "./app"
// Platform + client infrastructure re-exported from client-common
export {
  PlatformProvider,
  usePlatform,
  createAgentClient,
  AgentClientProvider,
  useAgentClient,
  type AgentClient,
  type AgentClientInstance,
  type Platform,
  type Storage,
  type Clipboard,
  type Notification,
  type Dialogs,
  type TerminalCapabilities,
  stopDisplayViewController,
} from "@magnitudedev/client-common"
export { createBrowserPlatform } from "./platform/browser"
export { DaemonConnectionError } from "./components/daemon-connection-error"
export { injectCssVars, generateCssVars, generateCssVarsString } from "./styles/generate-css-vars"
export { DiffHunk, type DiffHunkProps } from "./components/diff-hunk"
