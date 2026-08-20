import { Command } from "@commander-js/extra-typings"
import { registerDocsCommand } from "./commands/docs"
import { registerInteractiveCommand } from "./commands/interactive"
import { registerStopCommand } from "./commands/stop"
import { registerUpdateCommand } from "./commands/update"
import { CLI_VERSION } from "./version"

const program = new Command()
  .name("magnitude")
  .version(CLI_VERSION)

registerStopCommand(program)
registerUpdateCommand(program)
registerDocsCommand(program)
registerInteractiveCommand(program)

await program.parseAsync()
