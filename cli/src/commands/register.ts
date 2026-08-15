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
    // Cloud is disabled.
    // {
    //   id: "cloud",
    //   label: "cloud",
    //   description: "Manage Magnitude Cloud connection",
    // },
  ])
}
