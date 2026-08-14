import type { Command } from "@commander-js/extra-typings"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Exit, Option, Scope } from "effect"
import type { AuthSource } from "../state/cli-atoms"
import {
  developmentLaunchCommand,
  runInteractiveCommand,
  type InteractiveLaunchOptions,
} from "../runtime/interactive"
import { runHeadless } from "./headless"
import { makeTerminalPlatform } from "../platform/terminal"
import { makeCliEffectLoggingLayer } from "../platform/effect-logger"
import {
  flushProcessOutput,
  scheduleBoundedProcessExit,
} from "../utils/flush-process-output"
import { isDevelopmentBuild } from "../runtime/environment"

const resolveEnvAuth = (): AuthSource => {
  const envKey = process.env.MAGNITUDE_API_KEY
  return envKey && envKey.trim()
    ? { source: "env", key: envKey, envVarName: "MAGNITUDE_API_KEY" }
    : { source: "none" }
}

const createHeadlessPlatform = (options: InteractiveLaunchOptions) => async () => {
  const scope = await Effect.runPromise(Scope.make())
  try {
    const terminal = await Effect.runPromise(makeTerminalPlatform({
      launchCommand: developmentLaunchCommand(options),
      debug: options.debug,
      effectLoggingLayer: Option.some(makeCliEffectLoggingLayer({ debug: options.debug })),
    }).pipe(Scope.extend(scope)))
    return {
      ...terminal.platform,
      async shutdown() {
        try {
          return await terminal.platform.shutdown()
        } finally {
          await Effect.runPromise(Scope.close(scope, Exit.void))
        }
      },
    }
  } catch (error) {
    await Effect.runPromise(Scope.close(scope, Exit.void))
    throw error
  }
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

      if (opts.headless) {
        return Effect.runPromise(runHeadless({
          debug: options.debug,
          autopilot: opts.autopilot ?? false,
          ...(opts.prompt === undefined ? {} : { initialPrompt: opts.prompt }),
          sessionStart: options.sessionStart,
          disableShellSafeguards: options.sessionOptions.disableShellSafeguards ?? false,
          disableCwdSafeguards: options.sessionOptions.disableCwdSafeguards ?? false,
          ...(opts.atif === undefined ? {} : { atifPath: opts.atif }),
          ...(opts.goal === undefined ? {} : { goal: opts.goal }),
          solo: opts.solo ?? false,
          ...(opts.systemOverride === undefined ? {} : { systemOverride: opts.systemOverride }),
          setup: opts.setup ?? false,
        }, {
          createPlatform: createHeadlessPlatform(options),
          onTerminationSignal: (exitCode) => {
            Effect.runSync(scheduleBoundedProcessExit(exitCode))
          },
        })).then(async (exitCode) => {
          await Effect.runPromise(flushProcessOutput())
          Effect.runSync(scheduleBoundedProcessExit(exitCode))
        })
      }

      return Effect.runPromise(runInteractiveCommand(options).pipe(
        Effect.provide([BunContext.layer, FetchHttpClient.layer]),
        Effect.catchAll((error) => Effect.sync(() => {
          process.stderr.write(`Failed to start Magnitude: ${error.reason}\n`)
          return 1
        })),
      )).then((exitCode) => process.exit(exitCode))
    })
}
