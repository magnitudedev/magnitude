import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { fileURLToPath } from "node:url"
import { stringify } from "yaml"
import { Digest, InfrastructureFailure } from "../domain"
import { packageManager } from "../../../../package.json"

const Download = Schema.Struct({ url: Schema.Redacted(Schema.NonEmptyString.pipe(Schema.filter(value => {
  try { const url = new URL(value); return url.protocol === "https:" && !url.username && !url.password && !url.hash } catch { return false }
}))), sha256: Digest, bytes: Schema.Int.pipe(Schema.between(1, 1024 ** 3)) })
export const UbuntuInitialization = Schema.Struct({ adminUsername: Schema.String.pipe(Schema.pattern(/^[a-z][a-z0-9]{1,19}$/)),
  architecture: Schema.Literal("x64", "arm64"), runtime: Download, node: Download, rustup: Download })

/** cloud-init is privileged administrator configuration, never candidate-supplied code. */
export const ubuntuInitialization = (config: typeof UbuntuInitialization.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const setup = yield* fs.readFileString(fileURLToPath(new URL("../../infra/ubuntu-worker.sh", import.meta.url)))
  const configuration = yield* Schema.encode(Schema.parseJson(Schema.extend(UbuntuInitialization, Schema.Struct({ bunVersion: Schema.String }))))({
    ...config, bunVersion: packageManager.replace(/^bun@/, ""),
  })
  const text = `#cloud-config\n${stringify({ write_files: [
    { path: "/etc/magnitude-lab-initialization.json", owner: "root:root", permissions: "0600", content: configuration },
    { path: "/opt/magnitude-lab-initialize.sh", owner: "root:root", permissions: "0700", content: setup },
  ], runcmd: [["/bin/bash", "/opt/magnitude-lab-initialize.sh"]] })}`
  if (Buffer.byteLength(text) > 64 * 1024) return yield* new InfrastructureFailure({ operation: "ubuntu-initialization", message: "Generated cloud-init exceeds Azure's custom data limit" })
  return text
})
