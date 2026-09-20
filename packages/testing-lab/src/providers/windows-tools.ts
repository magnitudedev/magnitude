import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { fileURLToPath } from "node:url"
import { InitializationDownload } from "./linux-initialization"

/** Native Windows tooling is administrator-pinned, separate from candidate build inputs. */
export const WindowsToolDownloads = Schema.Struct({
  bun: InitializationDownload, node: InitializationDownload, git: InitializationDownload,
  powershell: InitializationDownload, ninja: InitializationDownload, cmake: InitializationDownload,
  nsis: InitializationDownload, uv: InitializationDownload, pi: InitializationDownload,
  opencode: InitializationDownload, tirith: InitializationDownload, rustup: InitializationDownload,
  vsBuildTools: InitializationDownload, vsChannel: InitializationDownload,
})
export const windowsToolDownloads = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  return yield* fs.readFileString(fileURLToPath(new URL("../../tools/windows-downloads.json", import.meta.url))).pipe(
    Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WindowsToolDownloads))),
  )
})

/** This prepares tooling only; it cannot emit a complete worker-readiness receipt. */
export const renderWindowsToolPreparation = (downloads: typeof WindowsToolDownloads.Type, setup: string) => Effect.gen(function* () {
  const json = yield* Schema.encode(Schema.parseJson(WindowsToolDownloads))(downloads)
  const script = Buffer.from(setup).toString("base64")
  const pins = Buffer.from(json).toString("base64")
  return String.raw`$ErrorActionPreference = 'Stop'
$directory = 'C:\MagnitudeLab\Preparation'
New-Item -ItemType Directory -Path $directory -Force | Out-Null
[IO.File]::WriteAllBytes((Join-Path $directory 'tools.json'), [Convert]::FromBase64String('${pins}'))
[IO.File]::WriteAllBytes((Join-Path $directory 'tools.ps1'), [Convert]::FromBase64String('${script}'))
& (Join-Path $directory 'tools.ps1') -ConfigurationFile (Join-Path $directory 'tools.json')
`
})
export const windowsToolPreparation = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const script = yield* fs.readFileString(fileURLToPath(new URL("../../infra/windows-tools.ps1", import.meta.url)))
  return yield* renderWindowsToolPreparation(yield* windowsToolDownloads, script)
})
