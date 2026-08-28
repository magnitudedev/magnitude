import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./interactive-runtime")

export const registerInteractiveCommand = (program: Command): void => {
  const interactiveCommand = program
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

  interactiveCommand.action((opts) =>
    loadRuntime().then(({ runInteractive }) => runInteractive(opts, false)))

  program
    .command("setup")
    .description("Interactive first time setup for installing a model and connecting it to a harness")
    .action(() => loadRuntime().then(({ runInteractive }) =>
      runInteractive(interactiveCommand.opts(), true)))
}
