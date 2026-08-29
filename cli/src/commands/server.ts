import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./server-runtime")

export const registerServiceCommand = (program: Command): void => {
  const service = program.command("service")
    .description("Manage the Magnitude local inference service")

  service.command("install")
    .description("Install and enable the per-user Magnitude service")
    .action(() => loadRuntime().then(({ runServiceInstall }) => runServiceInstall()))

  service.command("uninstall")
    .description("Stop and remove the per-user Magnitude service")
    .action(() => loadRuntime().then(({ runServiceUninstall }) => runServiceUninstall()))

  service.command("start")
    .description("Install and start the per-user Magnitude service")
    .action(() => loadRuntime().then(({ runServiceStart }) => runServiceStart()))

  service.command("stop")
    .description("Stop the Magnitude service without uninstalling it")
    .action(() => loadRuntime().then(({ runServiceStop }) => runServiceStop()))
  service.command("status")
    .description("Show installation and runtime status")
    .action(() => loadRuntime().then(({ runServiceStatus }) => runServiceStatus()))
}
