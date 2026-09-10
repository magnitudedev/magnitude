import { Effect, Schema, Stream } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import { createHash } from "node:crypto"
import { ACN_EXECUTABLE_NAME } from "./executables"

/** The macOS ACN artifact is a signed, notarized, stapled bundle whose main executable is the service. */
export const MACOS_APP_NAME = "Magnitude.app"
export const MACOS_BUNDLE_ID = "dev.magnitude.service"
export const MACOS_ACN_PATH = `${MACOS_APP_NAME}/Contents/MacOS/${ACN_EXECUTABLE_NAME}`

export const ArtifactDigest = Schema.String.pipe(
  Schema.pattern(/^[a-f0-9]{64}$/), Schema.brand("ArtifactDigest"),
)
export const SourceCommit = Schema.String.pipe(
  Schema.pattern(/^[a-f0-9]{40}$/), Schema.brand("SourceCommit"),
)

export const sha256File = (file: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const hash = yield* fs.stream(file).pipe(Stream.runFold(createHash("sha256"), (hash, bytes) => hash.update(bytes)))
  return hash.digest("hex")
})

export const acnExecutableRelativePath = (host: string): string =>
  host.startsWith("darwin-") ? MACOS_ACN_PATH : `bin/${ACN_EXECUTABLE_NAME}${host.startsWith("windows-") ? ".exe" : ""}`

export const MACOS_REQUIRED_FILES = [
  "Contents/Info.plist",
  `Contents/MacOS/${ACN_EXECUTABLE_NAME}`,
  "Contents/Resources/Magnitude.icns",
  "Contents/_CodeSignature/CodeResources",
] as const
