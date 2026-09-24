import { Argument, type Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./update-runtime")

export const registerUpdateCommand = (program: Command): void => {
  program
    .command("update")
    .description("Check, download or install application updates")
    .addArgument(new Argument("[action]", "install restarts Desktop; a running server must be stopped first").choices(["check", "status", "download", "install", "discard"]).default("check"))
    .action(action => loadRuntime().then(({ runUpdate }) => runUpdate(action)))
}
