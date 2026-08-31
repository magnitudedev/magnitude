import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./connections-runtime")

export const registerConnectionsCommand = (program: Command): void => {
  const connections = program.command("connections")
    .description("Manage Magnitude harness connections")

  connections.command("list")
    .description("List supported harnesses and installation status")
    .action(() => loadRuntime().then(({ listConnections }) => listConnections()))

  connections.command("add")
    .description("Connect every installed Magnitude model to a harness")
    .argument("<harness>", "Harness ID")
    .option("--set-current <model-id>", "Also select this Magnitude model in the harness")
    .option("--install-skill", "Install or refresh the Magnitude skill for this harness")
    .action((harness, options) => loadRuntime().then(({ addConnection }) =>
      addConnection(harness, options.setCurrent, options.installSkill === true)))

  connections.command("sync")
    .description("Reconcile configured harnesses with their Magnitude bindings")
    .argument("[harness]", "Harness ID")
    .action((harness) => loadRuntime().then(({ syncConnections }) =>
      syncConnections(harness)))

  connections.command("remove")
    .description("Remove a Magnitude harness connection")
    .argument("<harness>", "Harness ID")
    .action((harness) => loadRuntime().then(({ removeConnection }) =>
      removeConnection(harness)))
}
