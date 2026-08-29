import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./server-runtime")

export const registerServiceCommand = (program: Command): void => {
  const service = program.command("service")
    .description("Manage the Magnitude local inference service")

  service.command("start")
    .description("Install and start the per-user Magnitude service")
    .action(() => loadRuntime().then(({ runServiceStart }) => runServiceStart()))

  service.command("stop")
    .description("Stop and disable the per-user Magnitude service")
    .action(() => loadRuntime().then(({ runServiceStop }) => runServiceStop()))
}
