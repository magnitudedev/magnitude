import { Option, type Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./interactive-command-runtime")

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

  interactiveCommand.action((opts) => {
    const globals = interactiveCommand.optsWithGlobals() as { version?: boolean }
    return loadRuntime().then(({ runInteractiveCommand }) =>
      runInteractiveCommand(opts, globals))
  })

  const setupCommand = program
    .command("setup")
    .description("Interactive first time setup for installing a model and connecting it to a harness")
    .addOption(new Option("--host <harness>", "Return to the hosting harness after setup").choices(["pi"]).hideHelp())
    .action((opts) => {
      if (opts.host === "pi") {
        if (Object.values(interactiveCommand.opts()).some(value => value !== undefined)) {
          setupCommand.error("Pi setup cannot be combined with chat options")
        }
        return import("./pi-setup-runtime").then(({ runPiSetup }) => runPiSetup())
      }
      return loadRuntime().then(({ runSetupCommand }) =>
        runSetupCommand(interactiveCommand.opts()))
    })
}
