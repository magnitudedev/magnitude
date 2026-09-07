import type { Command } from "@commander-js/extra-typings"

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
    .option("--host-protocol", "Print the terminal-host protocol capability without starting setup")
    .option("--host <harness>", "Return setup to an existing Pi session")
    .option("--result-file <path>", "Write the hosted setup outcome to a private absolute path")
    .action((opts) => {
      const hosted = opts.host !== undefined || opts.resultFile !== undefined
      if (opts.hostProtocol && hosted) setupCommand.error("--host-protocol cannot be combined with hosted setup options")
      if (hosted && (opts.host !== "pi" || !opts.resultFile)) setupCommand.error("Hosted setup requires --host pi and --result-file <path>")
      if ((hosted || opts.hostProtocol) && Object.values(interactiveCommand.opts()).some(value => value !== undefined)) {
        setupCommand.error("Hosted setup cannot be combined with chat options")
      }
      return loadRuntime().then(({ runSetupCommand }) =>
        runSetupCommand(interactiveCommand.opts(), opts))
    })

}
