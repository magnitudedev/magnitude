import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./server-runtime")

export const registerServerCommand = (program: Command): void => {
  const server = program.command("server")
    .description("Manage the Magnitude local inference service")

  server.command("start")
    .description("Install and start the per-user Magnitude service")
    .action(() => loadRuntime().then(({ runServerStart }) => runServerStart()))

  server.command("stop")
    .description("Stop and disable the per-user Magnitude service")
    .action(() => loadRuntime().then(({ runServerStop }) => runServerStop()))
}
