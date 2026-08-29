import { Command } from "@commander-js/extra-typings"
import { registerDocsCommand } from "./commands/docs"
import { registerInteractiveCommand } from "./commands/interactive"
import { registerUpdateCommand } from "./commands/update"
import { registerServiceCommand } from "./commands/server"
import { registerInferenceCommands } from "./commands/inference"
import { registerConnectionsCommand } from "./commands/connections"
import { encodeErrorDocument, setOutputMode } from "./commands/output"

const program = new Command()
  .name("magnitude")
  .option("-v, --version", "Output the version")
  .option("--json", "Emit the command result as JSON")

const jsonRequested = process.argv.includes("--json")
program.configureOutput({
  writeErr: (message) => {
    process.stderr.write(jsonRequested
      ? `${JSON.stringify(encodeErrorDocument({
          code: "invalid_command",
          message: message.trim(),
          retryable: false,
        }))}\n`
      : message)
  },
})

program.hook("preAction", (_command, actionCommand) => {
  setOutputMode(actionCommand.optsWithGlobals().json === true)
})

registerServiceCommand(program)
registerInferenceCommands(program)
registerConnectionsCommand(program)
registerUpdateCommand(program)
registerDocsCommand(program)
registerInteractiveCommand(program)

await program.parseAsync()
