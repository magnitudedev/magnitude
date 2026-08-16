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
  onboardingModelSetupViewAtom,
} from "@magnitudedev/client-common";
import { CliApp, type SessionStart } from "./app";
import type { AuthSource } from "./state/cli-atoms";
import { getLastSessionId } from "./state/last-session";
import { CLI_VERSION } from "./version";
import { installGracefulShutdownHandlers } from "./utils/graceful-shutdown";
import { createTerminalPlatform, stopTerminalAcn } from "./platform/terminal";
import { makeCliEffectLoggingLayer } from "./platform/effect-logger";
import { Array as Arr, Effect, Exit, Option, Scope } from "effect";
import { registerDocsCommand } from "./commands/docs";
import {
  detectTerminalAppearance,
  installTerminalAppearanceRuntime,
} from "./platform/terminal-appearance";
import { terminalAppearanceAtom } from "./hooks/use-theme";

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

    // Headless mode is temporarily disabled while the CLI transitions to a
    // pure SDK/RPC client architecture. It needs a daemon-backed persistence
    // design before it can run again.
    if (opts.headless) {
      process.stderr.write(
        "Error: --headless is temporarily disabled. Use the TUI mode.\n"
      );
      process.exit(1);
    }

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
    Atom.runtime.addGlobalLayer(effectLoggingLayer);
    const platform = await createTerminalPlatform({
      launchCommand,
      debug: opts.debug === true,
      effectLoggingLayer: Option.some(effectLoggingLayer),
    });
    const initialAcnLifecycleState = await Effect.runPromise(
      platform.acnStartup.prepare
    );
    const agentClientTag = createAgentClient(platform.protocolLayer, {
      onboardingSetupInitiallyOpen: opts.setup ?? false,
    });
    await Effect.runPromise(Effect.exit(
      Registry.getResult(
        atomRegistry,
        onboardingModelSetupViewAtom(agentClientTag),
      ),
    ));
    const renderer = await createCliRenderer({
      exitOnCtrlC: false, // We handle Ctrl+C manually for two-tap exit
    });
    const terminalAppearance = await Effect.runPromise(
      detectTerminalAppearance(renderer)
    );
    atomRegistry.set(terminalAppearanceAtom, terminalAppearance);
    const terminalAppearanceScope = await Effect.runPromise(Scope.make());
    await Effect.runPromise(
      installTerminalAppearanceRuntime(renderer, atomRegistry).pipe(
        Effect.provideService(Scope.Scope, terminalAppearanceScope)
      )
    );
    renderer.once("destroy", () => {
      Effect.runFork(Scope.close(terminalAppearanceScope, Exit.void));
    });
    let modelExitNotice: string | undefined;

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

  program.parse();
}

main();
