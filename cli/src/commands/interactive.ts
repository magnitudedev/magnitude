import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./interactive-runtime")

export const registerInteractiveCommand = (program: Command): void => {
  const interactiveCommand = program
    .option(
      "--resume [id]",
      "Resume the most recent chat session or a specific session by ID",
    )
    .option("--prompt <text>", "Start session with an initial user message")
    .option("--atif <path>", "Write ATIF trajectory to the specified path")
    .option(
      "--system-override <text>",
      "Override system prompt with raw text",
    )

  interactiveCommand.action((opts) =>
    loadRuntime().then(({ runInteractive }) => runInteractive(opts, false)))

  program
    .command("setup")
    .description("Interactive first time setup for installing a model and connecting it to a harness")
    .action(() => loadRuntime().then(({ runInteractive }) =>
      runInteractive(interactiveCommand.opts(), true)))
}
