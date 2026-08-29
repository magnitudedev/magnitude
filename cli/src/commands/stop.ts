import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./stop-runtime")

export const registerStopCommand = (program: Command): void => {
  program
    .command("stop")
    .description("Stop the current Magnitude service and release its local models")
    .action(() => loadRuntime().then(({ runStop }) => runStop()))
}
