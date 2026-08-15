import { registerClientCommands } from "@magnitudedev/client-common"

export function registerCliCommands(): void {
  registerClientCommands([
    {
      id: "setup",
      label: "setup",
      description: "Open the onboarding setup screen",
    },
    {
      id: "models",
      label: "models",
      description: "Choose a ready model",
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
    {
      id: "load",
      label: "load",
      description: "Load the selected model when it is ready",
    },
    {
      id: "stop",
      label: "stop",
      description: "Cancel or stop the selected model",
    },
    // Cloud is disabled.
    // {
    //   id: "cloud",
    //   label: "cloud",
    //   description: "Manage Magnitude Cloud connection",
    // },
  ])
}
