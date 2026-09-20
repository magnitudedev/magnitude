import { Schema } from "effect"
import { Digest } from "../domain"
import hermes from "../../tools/hermes.json"

const Version = Schema.String.pipe(Schema.pattern(/^\d+\.\d+\.\d+$/))
const Download = Schema.Struct({ url: Schema.String.pipe(Schema.pattern(/^https:\/\/github\.com\/sheeki03\/tirith\/releases\/download\/v[0-9.]+\/tirith-(x86_64|aarch64)-unknown-linux-gnu\.tar\.gz$/)),
  sha256: Digest, bytes: Schema.Int.pipe(Schema.between(1, 128 * 1024 * 1024)) })
export const HermesInstallation = Schema.Struct({ version: Version,
  repository: Schema.Literal("https://github.com/NousResearch/hermes-agent.git"),
  commit: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/), Schema.brand("HermesCommit")),
  tirith: Schema.Struct({ version: Version, downloads: Schema.Struct({ x64: Download, arm64: Download }) }),
})
export const hermesInstallation = Schema.decodeUnknownSync(HermesInstallation)(hermes)
