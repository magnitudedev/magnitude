import { Argument, type Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./update-runtime")

export const registerUpdateCommand = (program: Command): void => {
  program
    .command("update")
    .description("Check, download or install desktop application updates")
    .addArgument(new Argument("[action]", "install stops the model and service before restarting").choices(["check", "status", "download", "install"]).default("check"))
    .action(action => loadRuntime().then(({ runUpdate }) => runUpdate(action)))
}
