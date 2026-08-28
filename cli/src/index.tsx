import { Command } from "@commander-js/extra-typings"
import { registerDocsCommand } from "./commands/docs"
import { registerInteractiveCommand } from "./commands/interactive"
import { registerStopCommand } from "./commands/stop"
import { registerUpdateCommand } from "./commands/update"
import { registerServerCommand } from "./commands/server"
import { registerInferenceCommands } from "./commands/inference"
import { registerConnectionsCommand } from "./commands/connections"
import { CLI_VERSION } from "./version"

const program = new Command()
  .name("magnitude")
  .version(CLI_VERSION, "-v, --version")

registerStopCommand(program)
registerServerCommand(program)
registerInferenceCommands(program)
registerConnectionsCommand(program)
registerUpdateCommand(program)
registerDocsCommand(program)
registerInteractiveCommand(program)

await program.parseAsync()
