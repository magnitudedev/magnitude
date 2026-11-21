import { Command } from "@commander-js/extra-typings"
import { registerApplicationCommand } from "./commands/application"
import { registerDocsCommand } from "./commands/docs"
import { registerUpdateCommand } from "./commands/update"
import { registerStatusCommand } from "./commands/status"
import { registerServeCommand } from "./commands/serve"
import { registerInferenceCommands } from "./commands/inference"
import { registerConnectionsCommand } from "./commands/connections"
import { registerHuggingFaceCommand } from "./commands/hugging-face-account"
import { admitsPkexecCommand } from "./startup/pkexec-admission"
import { ignoreClosedOutput } from "./startup/closed-output"

ignoreClosedOutput(process.stdout)

if (!admitsPkexecCommand(process.argv.slice(2), process.getuid?.(), process.env)) {
  process.stderr.write("Under pkexec, Magnitude runs only its update installer.\n")
  process.exit(1)
}

const program = new Command()
  .name("magnitude")
  .option("-v, --version", "Print the Magnitude version")

registerApplicationCommand(program)
registerStatusCommand(program)
// Release probe for the signed CLI's native database and runtime.
program.command("native-runtime-check", { hidden: true })
  .action(() => import("./commands/native-runtime-check").then(({ runNativeRuntimeCheck }) => runNativeRuntimeCheck()))
registerServeCommand(program)
// Run as root to convert a server set up before the CLI-only install: `--unit` by the package's install step,
// and with no option by the converted unit's ExecStartPre.
program.command("_convert-server", { hidden: true }).option("--unit")
  .action(options => import("./startup/convert-server").then(({ runConvertServer }) => runConvertServer(options.unit === true)))
registerInferenceCommands(program)
registerHuggingFaceCommand(program)
registerConnectionsCommand(program)
registerUpdateCommand(program)
registerDocsCommand(program)
program.command("_verify-windows-installation", { hidden: true }).argument("<offer>").argument("<artifact>").argument("<channel>").argument("[package]", "", "windows-exe")
  .action(async (offer, artifact, channel, windowsPackage) => {
    const { verifyInstallationDownload } = await import("./startup/verify-installation")
    await verifyInstallationDownload(offer, artifact, channel, windowsPackage)
  })
// Runs as root through sudo, or Polkit's installer wrapper. It validates the raw arguments itself instead of commander's parsing.
program.command("_install-application-update", { hidden: true }).argument("[arguments...]").allowUnknownOption().helpOption(false).action(async () => {
  const { runLinuxUpdateInstallation } = await import("./startup/linux-update-installation")
  await runLinuxUpdateInstallation(process.argv.slice(2))
})
program.command("_complete-application-update", { hidden: true }).action(async () => {
  const { runLinuxUpdateHandoff } = await import("./startup/linux-update-installation")
  await runLinuxUpdateHandoff()
})
program.command("_install-mac-application-update", { hidden: true }).argument("<request>").action(async (request: string) => {
  const { runMacApplicationInstallation } = await import("./startup/mac-installer")
  await runMacApplicationInstallation(request)
})
program.command("_install-mac-application", { hidden: true }).argument("<request>").action(async (request: string) => {
  const { runMacArchiveInstallation } = await import("./startup/mac-installer")
  await runMacArchiveInstallation(request)
})
program.command("_install-cli", { hidden: true }).action(async () => {
  const { runCliInstallation } = await import("./startup/install-cli")
  await runCliInstallation()
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
