import { Data, Effect, Schema } from "effect"
import { CLI_VERSION } from "../version"
import { runCommand } from "./output"

class InteractiveJsonRejected extends Data.TaggedError("InteractiveJsonRejected")<{
  readonly code: string
  readonly message: string
  readonly retryable: boolean
}> {}

export const runInteractiveCommand = (
  options: unknown,
  globals: { readonly json?: boolean; readonly version?: boolean },
): Promise<void> => {
  if (globals.version === true) {
    return runCommand({
      effect: Effect.succeed({ version: CLI_VERSION }),
      schema: Schema.Struct({ version: Schema.String }),
      render: ({ version }) => `${version}\n`,
    })
  }
  if (globals.json === true) {
    return runCommand({
      effect: Effect.fail(new InteractiveJsonRejected({
        code: "interactive_json_unsupported",
        message: "`--json` requires a non-interactive command",
        retryable: false,
      })),
      schema: Schema.Struct({}),
      render: () => "",
    })
  }
  return import("./interactive-runtime").then(({ runInteractive }) =>
    runInteractive(options as Parameters<typeof runInteractive>[0], false)).then(() => undefined)
}

export const runSetupCommand = (options: unknown): Promise<void> =>
  import("./interactive-runtime").then(({ runInteractive }) =>
    runInteractive(options as Parameters<typeof runInteractive>[0], true)).then(() => undefined)
