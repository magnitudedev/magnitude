import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { fileURLToPath } from "node:url"
import { stringify } from "yaml"
import { Digest, InfrastructureFailure } from "../domain"
import { packageManager } from "../../../../package.json"

export const InitializationDownload = Schema.Struct({ url: Schema.Redacted(Schema.NonEmptyString.pipe(Schema.filter(value => {
  try { const url = new URL(value); return url.protocol === "https:" && !url.username && !url.password && !url.hash } catch { return false }
}))), sha256: Digest, bytes: Schema.Int.pipe(Schema.between(1, 1024 ** 3)) })
export const LinuxDistribution = Schema.Union(
  Schema.Struct({ os: Schema.Literal("ubuntu"), version: Schema.Literal("24.04") }),
  Schema.Struct({ os: Schema.Literal("debian"), version: Schema.Literal("13") }),
  Schema.Struct({ os: Schema.Literal("fedora"), version: Schema.Literal("44") }),
)
export const LinuxInitialization = Schema.Struct({ distribution: LinuxDistribution, adminUsername: Schema.String.pipe(Schema.pattern(/^[a-z][a-z0-9]{1,19}$/)),
  architecture: Schema.Literal("x64", "arm64"), runtime: InitializationDownload, node: InitializationDownload, rustup: InitializationDownload })

/** cloud-init is privileged administrator configuration, never candidate-supplied code. */
export const linuxInitialization = (config: typeof LinuxInitialization.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const setup = yield* fs.readFileString(fileURLToPath(new URL("../../infra/linux-worker.sh", import.meta.url)))
  return yield* renderLinuxInitialization(config, setup)
})

export const renderLinuxInitialization = (config: typeof LinuxInitialization.Type, setup: string) => Effect.gen(function* () {
  const configuration = yield* Schema.encode(Schema.parseJson(Schema.extend(LinuxInitialization, Schema.Struct({ bunVersion: Schema.String }))))({
    ...config, bunVersion: packageManager.replace(/^bun@/, ""),
  })
  const text = `#cloud-config\n${stringify({ write_files: [
    { path: "/etc/magnitude-lab-initialization.json", owner: "root:root", permissions: "0600", content: configuration },
    { path: "/opt/magnitude-lab-initialize.sh", owner: "root:root", permissions: "0700", content: setup },
  ], runcmd: [["/bin/bash", "/opt/magnitude-lab-initialize.sh"]] })}`
  if (Buffer.byteLength(text) > 64 * 1024) return yield* new InfrastructureFailure({ operation: "linux-initialization", message: "Generated cloud-init exceeds Azure's custom data limit" })
  return text
})
