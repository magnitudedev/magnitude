import { resolve } from "path";
import { createCliRenderer } from "@opentui/core";
import { createRoot } from "@opentui/react";
import { Command } from "@commander-js/extra-typings";
import {
  Atom,
  Registry,
  RegistryContext,
  scheduleTask,
} from "@effect-atom/atom-react";
import {
  createAgentClient,
  AgentClientProvider,
  PlatformProvider,
  DisplayViewControllerProvider,
  deriveCliExitNotice,
  stopDisplayViewController,
  pushNotificationAtom,
} from "@magnitudedev/client-common";
import { CliApp, type SessionStart } from "./app";
import type { AuthSource } from "./state/cli-atoms";
import { getLastSessionId } from "./state/last-session";
import { CLI_VERSION } from "./version";
import { installGracefulShutdownHandlers } from "./utils/graceful-shutdown";
import { createTerminalPlatform, stopTerminalAcn } from "./platform/terminal";
import { makeCliEffectLoggingLayer } from "./platform/effect-logger";
import { runHeadless } from "./commands/headless";
import { Array as Arr, Effect, Option } from "effect";
import { registerDocsCommand } from "./commands/docs";
import {
  flushProcessOutput,
  scheduleBoundedProcessExit,
} from "./utils/flush-process-output";

/** One-time env-var auth resolution (spec §2.9) — not reactive. */
function resolveEnvAuth(): AuthSource {
  const envKey = process.env.MAGNITUDE_API_KEY;
  if (envKey && envKey.trim()) {
    return { source: "env", key: envKey, envVarName: "MAGNITUDE_API_KEY" };
  }
  return { source: "none" };
}

async function main() {
  const program = new Command()
    .name("magnitude")
    .version(CLI_VERSION)
    .option(
      "--resume [id]",
      "Resume the most recent chat session or a specific session by ID"
    )
    .option("--debug", "Enable debug mode with debug panel")
    .option("--autopilot", "Launch with autopilot enabled")
    .option("--prompt <text>", "Start session with an initial user message")
    .option("--headless", "Run in headless mode (no TUI, output to stdout)")
    .option(
      "--disable-shell-safeguards",
      "Disable shell command classification safeguards"
    )
    .option(
      "--disable-cwd-safeguards",
      "Disable working directory boundary safeguards"
    )
    .option("--atif <path>", "Write ATIF trajectory to the specified path")
    .option("--goal <objective>", "Start a goal for the session")
    .option("--solo", "Run without worker/task tools")
    .option(
      "--system-override <text>",
      "Override leader system prompt with raw text"
    )
    .option("--setup", "Rerun Local Models and Cloud Fallback setup");

  program
    .command("stop")
    .description("Stop the current Magnitude daemon and release its local models")
    .action(async () => {
      try {
        await stopTerminalAcn();
        process.stdout.write("Magnitude daemon stopped.\n");
      } catch (error) {
        process.stderr.write(`Failed to stop Magnitude daemon: ${String(error)}\n`);
        process.exitCode = 1;
      }
    });

  registerDocsCommand(program);

  program.action(async (opts) => {
    const sessionStart: SessionStart =
      opts.resume === undefined
        ? { _tag: "new" }
        : opts.resume === true
        ? { _tag: "latest" }
        : { _tag: "resume", sessionId: opts.resume };

    const isDev =
      import.meta.url.endsWith(".tsx") ||
      (process.argv[1]?.endsWith(".tsx") ?? false);
    const acnSourcePath = resolve(
      import.meta.dir,
      "..",
      "..",
      "packages",
      "acn",
      "src",
      "binary.ts"
    );
    const launchCommand: Option.Option<Arr.NonEmptyReadonlyArray<string>> = isDev
      ? Option.some([
          "bun",
          acnSourcePath,
          "serve",
          ...(opts.debug ? ["--debug"] : []),
        ])
      : Option.none();

    const atomRegistry = Registry.make({
      scheduleTask,
      defaultIdleTTL: 5_000,
    });
    const effectLoggingLayer = makeCliEffectLoggingLayer({
      debug: opts.debug === true,
      publishNotification: (notification) => {
        atomRegistry.set(pushNotificationAtom, notification);
      },
    });
    const createPlatform = () => createTerminalPlatform({
      launchCommand,
      debug: opts.debug === true,
      effectLoggingLayer: Option.some(effectLoggingLayer),
    });

    if (opts.headless) {
      const exitCode = await Effect.runPromise(runHeadless({
        debug: opts.debug === true,
        autopilot: opts.autopilot ?? false,
        ...(opts.prompt === undefined ? {} : { initialPrompt: opts.prompt }),
        sessionStart,
        disableShellSafeguards: opts.disableShellSafeguards ?? false,
        disableCwdSafeguards: opts.disableCwdSafeguards ?? false,
        ...(opts.atif === undefined ? {} : { atifPath: opts.atif }),
        ...(opts.goal === undefined ? {} : { goal: opts.goal }),
        solo: opts.solo ?? false,
        ...(opts.systemOverride === undefined ? {} : { systemOverride: opts.systemOverride }),
        setup: opts.setup ?? false,
      }, {
        createPlatform,
        onTerminationSignal: (exitCode) => {
          Effect.runSync(scheduleBoundedProcessExit(exitCode));
        },
      }));
      await Effect.runPromise(flushProcessOutput());
      // Allow just-late platform cleanup to run, but never let leaked runtime
      // handles make the bounded command retain the process indefinitely.
      Effect.runSync(scheduleBoundedProcessExit(exitCode));
      return;
    }

    Atom.runtime.addGlobalLayer(effectLoggingLayer);
    const platform = await createPlatform();
    const initialAcnLifecycleState = await Effect.runPromise(
      platform.acnStartup.prepare
    );
    const agentClientTag = createAgentClient(platform.protocolLayer);
    const renderer = await createCliRenderer({
      exitOnCtrlC: false, // We handle Ctrl+C manually for two-tap exit
    });
    let modelExitNotice: string | undefined;

    // Terminal background detection is handled by useTerminalBgDetection
    // inside the React tree (needs atom registry to write to themeAtom)

    installGracefulShutdownHandlers(
      renderer,
      async () => {
        const observation = await platform.shutdown();
        modelExitNotice = Option.getOrUndefined(
          deriveCliExitNotice(observation)
        );
        stopDisplayViewController();
      },
      () => {
        const notices: string[] = [];
        if (modelExitNotice) notices.push(modelExitNotice);
        const activeSessionId = getLastSessionId();
        if (activeSessionId) {
          notices.push(
            `Resume this session with:\nmagnitude --resume ${activeSessionId}`
          );
        }
        if (notices.length > 0) {
          process.stdout.write(`\n${notices.join("\n\n")}\n`);
        }
      }
    );

    createRoot(renderer).render(
      <PlatformProvider platform={platform}>
        <RegistryContext.Provider value={atomRegistry}>
          <AgentClientProvider tag={agentClientTag}>
            <DisplayViewControllerProvider>
              <CliApp
                sessionStart={sessionStart}
                initialPrompt={opts.prompt}
                goal={opts.goal}
                envAuth={resolveEnvAuth()}
                forceLocalInferenceSetup={opts.setup ?? false}
                initialAcnLifecycle={initialAcnLifecycleState}
                sessionOptions={{
                  disableShellSafeguards:
                    opts.disableShellSafeguards ?? false,
                  disableCwdSafeguards: opts.disableCwdSafeguards ?? false,
                  atifPath: opts.atif,
                  solo: opts.solo ?? false,
                  headless: false,
                  systemPromptOverride: opts.systemOverride,
                }}
              />
            </DisplayViewControllerProvider>
          </AgentClientProvider>
        </RegistryContext.Provider>
      </PlatformProvider>
    );
  });

  await program.parseAsync();
}

main();
