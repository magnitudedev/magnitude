import type { Command } from "@commander-js/extra-typings"

const loadRuntime = () => import("./hugging-face-account-runtime")

const JSON_HELP = "Print one machine-readable JSON document"

export const registerHuggingFaceCommand = (program: Command): void => {
  const huggingFace = program.command("hf")
    .description("Sign Magnitude in to Hugging Face for gated models, private repositories, and higher download limits")
  huggingFace.command("login")
    .description("Sign in with a code entered on any device; works over SSH and never opens a browser")
    .option("--json", "Print the sign-in code and then the result, each as one JSON line")
    .action((options) => loadRuntime().then(({ execute, signInToHuggingFace }) => execute(signInToHuggingFace(options))))
  huggingFace.command("status")
    .description("Show which Hugging Face account Magnitude is signed in to")
    .option("--json", JSON_HELP)
    .action((options) => loadRuntime().then(({ execute, showHuggingFaceStatus }) => execute(showHuggingFaceStatus(options))))
  huggingFace.command("logout")
    .description("Sign Magnitude out of Hugging Face and remove its saved sign-in")
    .option("--json", JSON_HELP)
    .action((options) => loadRuntime().then(({ execute, signOutOfHuggingFace }) => execute(signOutOfHuggingFace(options))))
}
