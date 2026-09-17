import { desktopApplication } from "../server/application"
import { runCommand } from "./output"

export const openApplication = () => runCommand({
  effect: desktopApplication.ensure("ShowWindow"),
  render: () => "Opened Magnitude.\n",
})
