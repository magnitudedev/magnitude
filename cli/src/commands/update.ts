import { Argument, type Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./update-runtime")

export const registerUpdateCommand = (program: Command): void => {
  program
    .command("update")
    .description("Check, download or install application updates")
    .addArgument(new Argument("[action]", "install restarts Desktop; a set-up server installs on its own when idle, and `magnitude serve` must be stopped first").choices(["check", "status", "download", "install", "discard"]).default("check"))
    .action(action => loadRuntime().then(({ runUpdate }) => runUpdate(action)))
}
