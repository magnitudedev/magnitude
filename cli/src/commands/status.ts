import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./status-runtime")

export const registerStatusCommand = (program: Command): void => {
  program.command("status")
    .description("Show the Magnitude owner, service, and active-model status")
    .action(() => loadRuntime().then(({ runStatus }) => runStatus()))
}
