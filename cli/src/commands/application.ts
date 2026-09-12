import type { Command } from "@commander-js/extra-typings"

export const registerApplicationCommand = (program: Command): void => {
  program.command("app").description("Open the Magnitude desktop application")
    .command("open").description("Show the Magnitude window")
    .action(() => import("./application-runtime").then(({ openApplication }) => openApplication()))
}
