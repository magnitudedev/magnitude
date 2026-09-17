import { runWindowsUpdateHandoff } from "../../../../cli/src/startup/windows-update-installation"

// Native acceptance compiles the same bootstrap used by the bundled headless CLI.
await runWindowsUpdateHandoff()
