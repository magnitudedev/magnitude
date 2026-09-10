import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./server-runtime")

export const registerServiceCommand = (program: Command): void => {
  // Release-only probe: the signed CLI must still load its native renderer and run Bun under Hardened Runtime.
  program.command("native-runtime-check", { hidden: true })
    .action(() => import("./native-runtime-check").then(({ runNativeRuntimeCheck }) => runNativeRuntimeCheck()))

  const service = program.command("service")
    .description("Manage the Magnitude background service")

  service.command("install")
    .description("Start Magnitude automatically at login without starting it now")
    .action(() => loadRuntime().then(({ runServiceInstall }) => runServiceInstall()))

  service.command("uninstall")
    .description("Stop Magnitude and remove it from login startup")
    .action(() => loadRuntime().then(({ runServiceUninstall }) => runServiceUninstall()))

  service.command("start")
    .description("Install and start the Magnitude service")
    .action(() => loadRuntime().then(({ runServiceStart }) => runServiceStart()))

  service.command("stop")
    .description("Stop Magnitude without removing it from login startup")
    .action(() => loadRuntime().then(({ runServiceStop }) => runServiceStop()))
  service.command("status")
    .description("Show Magnitude service and active-model status")
    .action(() => loadRuntime().then(({ runServiceStatus }) => runServiceStatus()))
}
