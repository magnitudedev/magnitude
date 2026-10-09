import { FileSystem } from "@effect/platform"
import { Effect } from "effect"
import { join } from "node:path"
import { renderUnixInstallationScript, renderWindowsInstallationScript } from "./installation-scripts"

/**
 * Produce a fresh static hosting directory with the two installation scripts; deploying it is a
 * separate release action. The scripts fetch their signed offers from the landing server's
 * counted `/api/installer` endpoint, so no offer is hosted statically.
 */
export const writeInstallationDistribution = (options: {
  readonly output: string
  readonly origin: string
  readonly appleTeam: string
  readonly windowsPublisher: string
  readonly publicKey: string
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const shell = yield* renderUnixInstallationScript({ origin: options.origin, appleTeam: options.appleTeam, publicKey: options.publicKey })
  const powershell = yield* renderWindowsInstallationScript({ origin: options.origin, publisher: options.windowsPublisher })
  // Render both before creating output; never overwrite an existing hosting tree.
  yield* fs.makeDirectory(options.output)
  yield* fs.writeFileString(join(options.output, "install.sh"), shell, { mode: 0o755 })
  yield* fs.writeFileString(join(options.output, "install.ps1"), powershell)
})
