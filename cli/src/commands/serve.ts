import type { Command } from "@commander-js/extra-typings"

export const registerServeCommand = (program: Command): void => {
  program.command("serve").description("Run Magnitude in the foreground")
    .action(() => import("./serve-runtime").then(({ runServe }) => runServe()))
}
