import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./update-runtime")

export const registerUpdateCommand = (program: Command): void => {
  program
    .command("update")
    .description("Update Magnitude with the package manager that installed it")
    .action(() => loadRuntime().then(({ runUpdate }) => runUpdate()))
}
