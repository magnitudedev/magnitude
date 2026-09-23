import { Command } from "@commander-js/extra-typings"
import { registerApplicationCommand } from "./commands/application"
import { registerDocsCommand } from "./commands/docs"
import { registerUpdateCommand } from "./commands/update"
import { registerStatusCommand } from "./commands/status"
import { registerServeCommand } from "./commands/serve"
import { registerInferenceCommands } from "./commands/inference"
import { registerConnectionsCommand } from "./commands/connections"

const program = new Command()
  .name("magnitude")
  .option("-v, --version", "Print the Magnitude version")

registerApplicationCommand(program)
registerStatusCommand(program)
// Release probe for the signed CLI's native database and runtime.
program.command("native-runtime-check", { hidden: true })
  .action(() => import("./commands/native-runtime-check").then(({ runNativeRuntimeCheck }) => runNativeRuntimeCheck()))
registerServeCommand(program)
registerInferenceCommands(program)
registerConnectionsCommand(program)
registerUpdateCommand(program)
registerDocsCommand(program)
program.command("_install-application-update", { hidden: true }).argument("<request>").action(async request => {
  const { runLinuxUpdateInstallation } = await import("./startup/linux-update-installation")
  await runLinuxUpdateInstallation(request)
})
program.command("_complete-application-update", { hidden: true }).action(async () => {
  const { runLinuxUpdateHandoff } = await import("./startup/linux-update-installation")
  await runLinuxUpdateHandoff()
})
program.command("_complete-mac-application-update", { hidden: true }).action(async () => {
  const { runMacUpdateHandoff } = await import("./startup/mac-update-installation")
  await runMacUpdateHandoff()
})
program.command("_complete-windows-application-update", { hidden: true }).action(async () => {
  const { runWindowsUpdateHandoff } = await import("./startup/windows-update-installation")
  await runWindowsUpdateHandoff()
})
program.action(async (options) => {
  if (options.version) {
    const { CLI_VERSION } = await import("./version")
    process.stdout.write(`${CLI_VERSION}\n`)
  } else {
    program.outputHelp()
  }
})

await program.parseAsync()
