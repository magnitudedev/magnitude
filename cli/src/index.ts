import { Command } from "@commander-js/extra-typings"
import { registerApplicationCommand } from "./commands/application"
import { registerDocsCommand } from "./commands/docs"
import { registerUpdateCommand } from "./commands/update"
import { registerServiceCommand } from "./commands/server"
import { registerInferenceCommands } from "./commands/inference"
import { registerConnectionsCommand } from "./commands/connections"

const program = new Command()
  .name("magnitude")
  .option("-v, --version", "Print the Magnitude version")

registerApplicationCommand(program)
registerServiceCommand(program)
registerInferenceCommands(program)
registerConnectionsCommand(program)
registerUpdateCommand(program)
registerDocsCommand(program)
program.action(async (options) => {
  if (options.version) {
    const { CLI_VERSION } = await import("./version")
    process.stdout.write(`${CLI_VERSION}\n`)
  } else {
    program.outputHelp()
  }
})

await program.parseAsync()
