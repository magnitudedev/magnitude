import { registerClientCommands } from "@magnitudedev/client-common"

/** Registers the local-model surfaces owned by the web presentation. */
export function registerWebCommands(): void {
  registerClientCommands([
    { id: "setup", label: "setup", description: "Open local model setup" },
    {
      id: "models",
      label: "models",
      description: "Choose a ready local model",
    },
    {
      id: "catalog",
      label: "catalog",
      description: "Find and download local models",
    },
    {
      id: "hardware",
      label: "hardware",
      description: "Inspect local inference hardware",
    },
  ])
}
