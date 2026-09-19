import { FileSystem } from "@effect/platform"
import { ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"
import { Effect, Option, Schema } from "effect"
import { randomUUID } from "node:crypto"
import { join } from "node:path"
import { releaseUrl } from "../../release/src/acquisition"
import { downloadObject } from "./artifact-store"
import { Digest, InfrastructureFailure, Target } from "./domain"

const failure = (message: string) => new InfrastructureFailure({ operation: "runtime-release", message })

/** Publish exact admitted runtime archives through ordinary release acquisition, without a
 * development installation override. The listener and its verified copies share the run scope. */
export const runtimeRelease = (release: typeof ReleaseManifestSchema.Type, host: Target["artifactHost"]) => Effect.gen(function* () {
  yield* validateReleaseManifest(release).pipe(Effect.mapError(error => failure(error.message)))
  const artifacts = release.artifacts.filter(artifact => Option.contains(artifact.host, host)
    && (artifact.kind === "icn-base" || artifact.kind === "icn-backend"))
  if (artifacts.filter(artifact => artifact.kind === "icn-base").length !== 1) {
    return yield* failure("Admitted release must include exactly one inference base for this host")
  }
  const fs = yield* FileSystem.FileSystem
  const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-lab-runtime-" })
  const prefix = `/${randomUUID()}`
  // releaseUrl is the production URL contract, including its scoped package tag encoding.
  const route = (name: string) => new URL(releaseUrl(`http://127.0.0.1${prefix}`, release.version, name)).pathname
  const files = new Map<string, { readonly path: string; readonly bytes: number; readonly digest: string }>()
  for (const artifact of artifacts) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._+-]*$/.test(artifact.filename) || artifact.filename === "magnitude-release.json" || files.has(route(artifact.filename))) {
      return yield* failure("Runtime archives must have unique safe basenames")
    }
    const path = join(directory, artifact.filename)
    yield* downloadObject(Digest.make(artifact.sha256), path)
    if (Number((yield* fs.stat(path)).size) !== artifact.bytes) return yield* failure("Runtime archive length differs from admitted manifest")
    yield* fs.chmod(path, 0o400)
    files.set(route(artifact.filename), { path, bytes: artifact.bytes, digest: artifact.sha256 })
  }
  const manifest = yield* Schema.encode(Schema.parseJson(ReleaseManifestSchema))(release)
  const manifestPath = route("magnitude-release.json")
  const server = yield* Effect.acquireRelease(Effect.try({ try: () => Bun.serve({ hostname: "127.0.0.1", port: 0,
    fetch(request) {
      const url = new URL(request.url)
      if (!["GET", "HEAD"].includes(request.method) || url.search) return new Response(null, { status: 404 })
      if (url.pathname === manifestPath) return new Response(request.method === "HEAD" ? null : manifest,
        { headers: { "content-type": "application/json", "content-length": String(Buffer.byteLength(manifest)), "cache-control": "no-store" } })
      const file = files.get(url.pathname)
      if (!file) return new Response(null, { status: 404 })
      const etag = `"${file.digest}"`
      const range = request.method === "GET" && (!request.headers.has("if-range") || request.headers.get("if-range") === etag)
        ? request.headers.get("range") : null
      let start = 0, end = file.bytes - 1
      if (range) {
        const match = /^bytes=(\d+)-(\d*)$/.exec(range)
        start = match ? Number(match[1]) : NaN
        end = match?.[2] ? Math.min(Number(match[2]), end) : end
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start > end) {
          return new Response(null, { status: 416, headers: { "content-range": `bytes */${file.bytes}` } })
        }
      }
      return new Response(request.method === "HEAD" ? null : Bun.file(file.path).slice(start, end + 1), { status: range ? 206 : 200, headers: {
        "content-type": "application/octet-stream", "content-length": String(end - start + 1), etag, "cache-control": "no-store", "accept-ranges": "bytes",
        ...(range ? { "content-range": `bytes ${start}-${end}/${file.bytes}` } : {}),
      } })
    },
  }), catch: () => failure("Could not start private runtime release listener") }), server => Effect.promise(() => server.stop(true)))
  return { baseUrl: `http://127.0.0.1:${server.port}${prefix}`, artifacts }
}).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : failure(error.message)))

/** App-only manifests use their ordinary released runtime. A manifest that includes native
 * artifacts must use its own admitted graph, never an ambient development installation. */
export const runtimeEnvironment = (release: typeof ReleaseManifestSchema.Type, host: Target["artifactHost"], environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  const ordinary = Object.fromEntries(Object.entries(environment).filter(([key]) => key.toUpperCase() !== "MAGNITUDE_ICN_PATH"))
  if (!release.artifacts.some(artifact => artifact.kind === "icn-base" || artifact.kind === "icn-backend")) return ordinary
  const origin = yield* runtimeRelease(release, host)
  const base = Object.fromEntries(Object.entries(ordinary).filter(([key]) => key.toUpperCase() !== "MAGNITUDE_RELEASE_BASE_URL"))
  const prepared: Record<string, string> = { ...base, MAGNITUDE_RELEASE_BASE_URL: origin.baseUrl }
  return prepared
})
