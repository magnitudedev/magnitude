import { Argument, type Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./update-runtime")

export const registerUpdateCommand = (program: Command): void => {
  program
    .command("update")
    .description("Check, download or install application updates")
    .addArgument(new Argument("[action]", "install restarts the desktop app; a running `magnitude serve` installs on its own once idle, or stop it to install now").choices(["check", "status", "download", "install", "discard"]).default("check"))
    .action(action => loadRuntime().then(({ runUpdate }) => runUpdate(action)))
}
