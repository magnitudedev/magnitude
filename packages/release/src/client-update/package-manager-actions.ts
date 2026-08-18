import { Option } from "effect"
import { CLI_PACKAGE_NAME } from "../contracts"
import type { InstallMethod, PackageManager } from "./install-context"

/**
 * The package-manager command that updates an installed client. The displayed
 * command and the executed command both come from this structured value.
 */
export interface UpdateAction {
  readonly method: PackageManager
  readonly command: string
  readonly args: readonly [string, ...string[]]
}

export const updateActionFor = (
  method: InstallMethod,
): Option.Option<UpdateAction> => {
  switch (method) {
    case "npm":
      return Option.some({
        method,
        command: "npm",
        args: ["install", "-g", CLI_PACKAGE_NAME],
      })
    case "bun":
      return Option.some({
        method,
        command: "bun",
        args: ["install", "-g", CLI_PACKAGE_NAME],
      })
    case "pnpm":
      return Option.some({
        method,
        command: "pnpm",
        args: ["add", "-g", CLI_PACKAGE_NAME],
      })
    case "other":
      return Option.none()
  }
}

export const updateCommandString = (action: UpdateAction): string =>
  [action.command, ...action.args].join(" ")
