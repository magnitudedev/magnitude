import type { Command } from "@commander-js/extra-typings"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import type { AuthSource } from "../state/cli-atoms"
import {
  runInteractiveCommand,
  type InteractiveLaunchOptions,
} from "../runtime/interactive"
import { isDevelopmentBuild } from "../runtime/environment"

const resolveEnvAuth = (): AuthSource => {
  const envKey = process.env.MAGNITUDE_API_KEY
  return envKey && envKey.trim()
    ? { source: "env", key: envKey, envVarName: "MAGNITUDE_API_KEY" }
    : { source: "none" }
}

export const registerInteractiveCommand = (program: Command): void => {
  program
    .option(
      "--resume [id]",
      "Resume the most recent chat session or a specific session by ID",
    )
    .option("--debug", "Enable debug mode with debug panel")
    .option("--autopilot", "Launch with autopilot enabled")
    .option("--prompt <text>", "Start session with an initial user message")
    .option("--headless", "Run in headless mode (no TUI, output to stdout)")
    .option(
      "--disable-shell-safeguards",
      "Disable shell command classification safeguards",
    )
    .option(
      "--disable-cwd-safeguards",
      "Disable working directory boundary safeguards",
    )
    .option("--atif <path>", "Write ATIF trajectory to the specified path")
    .option("--goal <objective>", "Start a goal for the session")
    .option("--solo", "Run without worker/task tools")
    .option(
      "--system-override <text>",
      "Override leader system prompt with raw text",
    )
    .option("--setup", "Rerun Local Models and Cloud Fallback setup")
    .action((opts) => {
      if (opts.headless) {
        process.stderr.write(
          "Error: --headless is temporarily disabled. Use the TUI mode.\n",
        )
        process.exit(1)
      }

      const options: InteractiveLaunchOptions = {
        debug: opts.debug === true,
        setup: opts.setup ?? false,
        developmentBuild: isDevelopmentBuild(),
        sessionStart: opts.resume === undefined
          ? { _tag: "new" }
          : opts.resume === true
            ? { _tag: "latest" }
            : { _tag: "resume", sessionId: opts.resume },
        initialPrompt: opts.prompt,
        goal: opts.goal,
        envAuth: resolveEnvAuth(),
        sessionOptions: {
          disableShellSafeguards: opts.disableShellSafeguards ?? false,
          disableCwdSafeguards: opts.disableCwdSafeguards ?? false,
          atifPath: opts.atif,
          solo: opts.solo ?? false,
          headless: false,
          systemPromptOverride: opts.systemOverride,
        },
      }

      return Effect.runPromise(runInteractiveCommand(options).pipe(
        Effect.provide([BunContext.layer, FetchHttpClient.layer]),
        Effect.catchAll((error) => Effect.sync(() => {
          const reason = "reason" in error ? error.reason : String(error)
          process.stderr.write(`Failed to start Magnitude: ${reason}\n`)
          return 1
        })),
      )).then((exitCode) => process.exit(exitCode))
    })
}
