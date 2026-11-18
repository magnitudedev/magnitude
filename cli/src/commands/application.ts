import type { Command } from "@commander-js/extra-typings"

export const registerApplicationCommand = (program: Command): void => {
  program.command("app").description("Open the Magnitude app")
    .command("open").description("Show the Magnitude window, or with the CLI only, open the app in a browser")
    .action(() => import("./application-runtime").then(({ openApplication }) => openApplication()))
}
