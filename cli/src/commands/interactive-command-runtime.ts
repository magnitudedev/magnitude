import { Effect } from "effect"
import { CLI_VERSION } from "../version"
import { runCommand } from "./output"
import { HostedSetupCapability, HOSTED_SETUP_PROTOCOL_VERSION } from "@magnitudedev/client-common/harness-connections/hosted-setup"
import { Schema } from "effect"

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

export const runSetupCommand = (
  options: unknown,
  host: { readonly hostProtocol?: boolean; readonly host?: string; readonly resultFile?: string } = {},
): Promise<void> => {
  if (host.hostProtocol) return runCommand({
    effect: Schema.encode(Schema.parseJson(HostedSetupCapability))({ protocolVersion: HOSTED_SETUP_PROTOCOL_VERSION }),
    render: value => `${value}\n`,
  })
  return import("./interactive-runtime").then(({ runInteractive }) =>
    runInteractive(options as Parameters<typeof runInteractive>[0], true, host.resultFile)).then(() => undefined)
}
