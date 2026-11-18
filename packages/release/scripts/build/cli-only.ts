import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Option, Schema } from "effect"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { ACN_EXECUTABLE_NAME } from "../../src/executables"
import { MACOS_BUNDLE_ID } from "../../src/macos-app"
import { cliOnlyPayloadPaths, type ReleaseHost } from "../../src/targets"
import { notarizeAppleUnit } from "../apple/distribution"
import { signAppleCode } from "../apple/signing"
import type { ArchiveSource } from "./common"
import { desktopUpdateTrustFiles } from "./desktop"
import { signWindowsCode } from "./windows-signing"

export class CliOnlyPayloadFailed extends Schema.TaggedError<CliOnlyPayloadFailed>()("CliOnlyPayloadFailed", { message: Schema.String }) {}

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..")

interface AppleSigning {
  readonly identifier: string
  readonly profile: "native" | "bun" | "library"
}

interface PayloadFile {
  readonly path: string
  readonly source: string
  readonly mode: number
  readonly sign: Option.Option<AppleSigning>
}

/**
 * The CLI-only payload's files and where each comes from: the host's built CLI and service, the native
 * helpers the desktop bundles, and the update trust the desktop build generated. The CLI arrives signed;
 * on macOS everything else native is signed here with the identities the desktop and service use.
 */
export const cliOnlyPayloadFiles = (host: ReleaseHost, inputs: { readonly cli: string; readonly service: string }): readonly PayloadFile[] => {
  const platform = host.id.startsWith("darwin-") ? "darwin" : host.id.startsWith("linux-") ? "linux" : "win32"
  const native = join(root, "packages/daemon-management/dist/native", `${platform}-${host.id.includes("arm64") ? "arm64" : "x64"}`)
  const apple = (signing: AppleSigning) => platform === "darwin" ? Option.some(signing) : Option.none<AppleSigning>()
  return [
    { path: `magnitude${host.executableExtension}`, source: inputs.cli, mode: 0o755, sign: Option.none() },
    { path: `${ACN_EXECUTABLE_NAME}${host.executableExtension}`, source: inputs.service, mode: 0o755, sign: apple({ identifier: MACOS_BUNDLE_ID, profile: "bun" }) },
    { path: "desktop-host.node", source: join(native, "desktop-host.node"), mode: 0o755, sign: apple({ identifier: "dev.magnitude.desktop-host", profile: "library" }) },
    platform === "win32"
      ? { path: "magnitude-launcher.exe", source: join(native, "magnitude-launcher.exe"), mode: 0o755, sign: Option.none() }
      : { path: "magnitude-command", source: join(native, "magnitude-command"), mode: 0o755, sign: apple({ identifier: "dev.magnitude.command", profile: "native" }) },
    { path: "update-configuration.json", source: desktopUpdateTrustFiles.configuration, mode: 0o644, sign: Option.none() },
    { path: "update-trust.json", source: desktopUpdateTrustFiles.trust, mode: 0o644, sign: Option.none() },
    { path: "LICENSE", source: join(root, "LICENSE"), mode: 0o644, sign: Option.none() },
  ]
}

/**
 * Copy the payload into `stage` and sign it: Apple code signatures and one notarization on macOS,
 * Authenticode for the native helpers on Windows (the CLI and service are already signed there).
 */
export const stageCliOnlyPayload = (options: {
  readonly host: ReleaseHost
  readonly cli: string
  readonly service: string
  readonly stage: string
  readonly output: string
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const files = cliOnlyPayloadFiles(options.host, options)
  const declared = cliOnlyPayloadPaths(options.host.id)
  const paths = files.map(file => file.path).sort()
  if (paths.length !== declared.length || paths.some((path, index) => path !== declared[index])) {
    return yield* new CliOnlyPayloadFailed({ message: `${options.host.id} CLI-only payload differs from its declared files` })
  }
  yield* fs.remove(options.stage, { recursive: true, force: true })
  yield* fs.makeDirectory(options.stage, { recursive: true })
  const sources: ArchiveSource[] = []
  for (const file of files) {
    const staged = join(options.stage, file.path)
    yield* fs.copyFile(file.source, staged)
    yield* fs.chmod(staged, file.mode)
    if (Option.isSome(file.sign)) yield* signAppleCode(staged, file.sign.value.identifier, file.sign.value.profile)
    sources.push({ path: file.path, source: staged, mode: file.mode })
  }
  if (options.host.id === "windows-x64-msvc") {
    yield* Effect.forEach(["desktop-host.node", "magnitude-launcher.exe"], name => signWindowsCode(join(options.stage, name)), { discard: true })
  }
  const notarization = options.host.id.startsWith("darwin-")
    ? yield* notarizeAppleUnit("cli-only", options.output, [
      ...files.filter(file => file.mode === 0o755).map(file => join(options.stage, file.path)),
      join(root, "bin/apple-inputs/acn"),
    ])
    : Option.none()
  return { sources, notarization }
})
