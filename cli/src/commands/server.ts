import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./server-runtime")

export const registerServerCommand = (program: Command): void => {
  const server = program.command("server").description("Run Magnitude as a service that starts at boot")
  server.command("setup").description("Set up Magnitude to run as a service on this machine")
    .action(() => loadRuntime().then(({ runServerSetup }) => runServerSetup()))
  server.command("remove").description("Stop the Magnitude service and remove it, keeping its data")
    .action(() => loadRuntime().then(({ runServerRemove }) => runServerRemove()))
  // Root steps run through sudo by full path. They validate the raw arguments themselves.
  for (const name of ["_server-install", "_server-remove"]) {
    program.command(name, { hidden: true }).argument("[arguments...]").allowUnknownOption().helpOption(false)
      .action(() => loadRuntime().then(({ runServerRootStep }) => runServerRootStep(process.argv.slice(2))))
  }
}
