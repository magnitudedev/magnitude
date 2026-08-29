import { Command } from "@commander-js/extra-typings"
import { registerDocsCommand } from "./commands/docs"
import { registerInteractiveCommand } from "./commands/interactive"
import { registerUpdateCommand } from "./commands/update"
import { registerServiceCommand } from "./commands/server"
import { registerInferenceCommands } from "./commands/inference"
import { registerConnectionsCommand } from "./commands/connections"
import { registerStopCommand } from "./commands/stop"
import { CLI_VERSION } from "./version"

const program = new Command()
  .name("magnitude")
  .version(CLI_VERSION, "-v, --version")

registerServiceCommand(program)
registerStopCommand(program)
registerInferenceCommands(program)
registerConnectionsCommand(program)
registerUpdateCommand(program)
registerDocsCommand(program)
registerInteractiveCommand(program)

await program.parseAsync()
