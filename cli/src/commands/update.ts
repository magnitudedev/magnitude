import { Argument, type Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./update-runtime")

export const registerUpdateCommand = (program: Command): void => {
  program
    .command("update")
    .description("Check, download or install application updates")
    .addArgument(new Argument("[action]", "install: the desktop app restarts into the update; a CLI-only `magnitude serve` installs it once idle; with nothing running it installs now").choices(["check", "status", "download", "install", "discard"]).default("check"))
    .action(action => loadRuntime().then(({ runUpdate }) => runUpdate(action)))
}
