import { Command } from "@commander-js/extra-typings"
import { registerDocsCommand } from "./commands/docs"
import { registerInteractiveCommand } from "./commands/interactive"
import { registerUpdateCommand } from "./commands/update"
import { registerServiceCommand } from "./commands/server"
import { registerInferenceCommands } from "./commands/inference"
import { registerConnectionsCommand } from "./commands/connections"
import { parseJsonCommand, requestedJsonCommand } from "./commands/json-command-line"

const program = new Command()
  .name("magnitude")
  .option("-v, --version", "Print the Magnitude version")

registerServiceCommand(program)
registerInferenceCommands(program)
registerConnectionsCommand(program)
registerUpdateCommand(program)
registerDocsCommand(program)
registerInteractiveCommand(program)

const args = process.argv.slice(2)
const jsonCommand = requestedJsonCommand(args)
if (jsonCommand === undefined) {
  await program.parseAsync()
} else {
  await parseJsonCommand(program, args, jsonCommand)
}
