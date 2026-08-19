import { CLI_PACKAGE_NAME } from "../contracts"
import type { PackageManager } from "./install-context"

/**
 * The package-manager command that updates an installed client to an exact
 * version. The displayed command and the executed command both come from this
 * structured value; pinning the version keeps the offer and the installation
 * identical, and keeps prerelease clients on their own channel — an unpinned
 * install would resolve `latest` at execution time.
 */
export interface UpdateAction {
  readonly method: PackageManager
  readonly command: string
  readonly args: readonly [string, ...string[]]
}

export const updateActionFor = (
  method: PackageManager,
  version: string,
): UpdateAction => {
  const packageSpec = `${CLI_PACKAGE_NAME}@${version}`
  switch (method) {
    case "npm":
      return { method, command: "npm", args: ["install", "-g", packageSpec] }
    case "bun":
      return { method, command: "bun", args: ["install", "-g", packageSpec] }
    case "pnpm":
      return { method, command: "pnpm", args: ["add", "-g", packageSpec] }
  }
}

export const updateCommandString = (action: UpdateAction): string =>
  [action.command, ...action.args].join(" ")
