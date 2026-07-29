import {
  releaseUrl,
  validateReleaseManifestBytes,
} from "../src/acquisition"
import { Effect } from "effect"

const version = process.env.MAGNITUDE_RELEASE_VERSION?.trim()
const sourceCommit = process.env.MAGNITUDE_SOURCE_COMMIT?.trim()
if (!version || !sourceCommit) {
  throw new Error("release version and source commit are required")
}
const baseUrl = "https://github.com/magnitudedev/magnitude/releases/download"
const download = async (name: string, maximum: number): Promise<Uint8Array> => {
  const response = await fetch(releaseUrl(baseUrl, version, name), {
    signal: AbortSignal.timeout(30_000),
  })
  if (!response.ok) throw new Error(`${name} returned HTTP ${response.status}`)
  const bytes = new Uint8Array(await response.arrayBuffer())
  if (bytes.byteLength > maximum) throw new Error(`${name} exceeds its size bound`)
  return bytes
}
const manifest = await download("magnitude-release.json", 16 * 1024 * 1024)
const release = await Effect.runPromise(
  validateReleaseManifestBytes(manifest),
)
if (
  release.manifest.version !== version ||
  release.manifest.sourceCommit !== sourceCommit
) {
  throw new Error("public release does not match the released source")
}
