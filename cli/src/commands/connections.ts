import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./connections-runtime")

export const registerConnectionsCommand = (program: Command): void => {
  const connections = program.command("connections")
    .description("Connect Magnitude models to agent harnesses")

  connections.command("list")
    .description("List supported harnesses and their Magnitude status")
    .action(() => loadRuntime().then(({ listConnections }) => listConnections()))

  connections.command("add")
    .description("Connect installed Magnitude models to a harness")
    .argument("<harness>", "Harness ID")
    .option("--set-model <model-id>", "Also select this Magnitude model in the harness")
    .option("--install-skill", "Install or refresh the Magnitude skill for this harness")
    .action((harness, options) => loadRuntime().then(({ addConnection }) =>
      addConnection(harness, options.setModel, options.installSkill === true)))

  connections.command("sync")
    .description("Refresh configured harnesses with installed Magnitude models")
    .argument("[harness]", "Harness ID")
    .action((harness) => loadRuntime().then(({ syncConnections }) =>
      syncConnections(harness)))

  connections.command("remove")
    .description("Disconnect Magnitude from a harness")
    .argument("<harness>", "Harness ID")
    .action((harness) => loadRuntime().then(({ removeConnection }) =>
      removeConnection(harness)))
}
