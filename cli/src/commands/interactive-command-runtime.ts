import { Effect } from "effect"
import { CLI_VERSION } from "../version"
import { runCommand } from "./output"

export const runInteractiveCommand = (
  options: unknown,
  globals: { readonly version?: boolean },
): Promise<void> => {
  if (globals.version === true) {
    return runCommand({
      effect: Effect.succeed({ version: CLI_VERSION }),
      render: ({ version }) => `${version}\n`,
    })
  }
  return import("./interactive-runtime").then(({ runInteractive }) =>
    runInteractive(options as Parameters<typeof runInteractive>[0], false)).then(() => undefined)
}

export const runSetupCommand = (options: unknown): Promise<void> =>
  import("./interactive-runtime").then(({ runInteractive }) =>
    runInteractive(options as Parameters<typeof runInteractive>[0], true)).then(() => undefined)
