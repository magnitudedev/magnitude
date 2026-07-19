import { registerClientCommands } from "@magnitudedev/client-common"

export function registerCliCommands(): void {
  registerClientCommands([
    {
      id: "cloud",
      label: "cloud",
      description: "Connect Magnitude Cloud models",
    },
  ])
}
