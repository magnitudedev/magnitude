import { Console, Context, Effect, Layer, Option, Schedule, Schema, Stream } from "effect"
import { Command, FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { join } from "node:path"

export class UploadHttpError extends Schema.TaggedError<UploadHttpError>()("UploadHttpError", {
  status: Schema.Number,
  message: Schema.String,
}) {}

export class UploadTransportError extends Schema.TaggedError<UploadTransportError>()("UploadTransportError", {
  message: Schema.String,
}) {}

class InvalidUploadDraft extends Schema.TaggedError<InvalidUploadDraft>()("InvalidUploadDraft", {
  message: Schema.String,
}) {}

const Asset = Schema.Struct({
  id: Schema.Number,
  name: Schema.String,
  size: Schema.Number,
  state: Schema.String,
  digest: Schema.optionalWith(Schema.NullOr(Schema.String), { as: "Option", exact: true }),
})

const Draft = Schema.Struct({
  draft: Schema.Boolean,
  tag_name: Schema.String,
  target_commitish: Schema.String,
})

export interface UploadFile {
  readonly name: string
  readonly path: string
  readonly bytes: number
  readonly sha256: string
}

export interface ReleaseUploadTransport {
  readonly upload: (url: string, token: string, file: UploadFile) => Effect.Effect<void, UploadHttpError | UploadTransportError>
}
export const ReleaseUploadTransport = Context.GenericTag<ReleaseUploadTransport>("ReleaseUploadTransport")

const TransferMetrics = Schema.Struct({
  http_code: Schema.Number,
  size_upload: Schema.Number,
  speed_upload: Schema.Number,
  time_total: Schema.Number,
})

export const CurlReleaseUploadTransport = Layer.succeed(ReleaseUploadTransport, {
  upload: (url, token, file) => Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-upload-" })
    const headersPath = join(temporary, "request.headers")
    const responsePath = join(temporary, "response.json")
    const responseHeadersPath = join(temporary, "response.headers")
    // Keep credentials out of process arguments and transfer logs.
    yield* fs.writeFileString(headersPath, `Authorization: Bearer ${token}\nAccept: application/vnd.github+json\nX-GitHub-Api-Version: 2022-11-28\nContent-Type: application/octet-stream\nConnection: close\n`, { mode: 0o600 })
    const child = yield* Command.make("curl", "--http1.1", "--show-error",
      "--connect-timeout", "15", "--max-time", "180", "--speed-time", "30", "--speed-limit", "1024",
      "--header", `@${headersPath}`, "--request", "POST", "--upload-file", file.path,
      "--output", responsePath, "--dump-header", responseHeadersPath,
      "--write-out", "%{json}", url,
    ).pipe(Command.stderr("inherit"), Command.start)
    const [metricsText, code] = yield* Effect.all([
      child.stdout.pipe(Stream.decodeText(), Stream.runFold("", (text, chunk) => text + chunk)),
      child.exitCode,
    ], { concurrency: "unbounded" })
    const metrics = yield* Schema.decodeUnknown(Schema.parseJson(TransferMetrics))(metricsText)
    yield* Console.log(`Transfer ${file.name}: sent ${metrics.size_upload}/${file.bytes} bytes, HTTP ${metrics.http_code}, ${metrics.time_total}s, ${Math.round(metrics.speed_upload)} bytes/s, curl exit ${code}`)
    const body = yield* fs.readFileString(responsePath).pipe(Effect.orElseSucceed(() => ""))
    const responseHeaders = yield* fs.readFileString(responseHeadersPath).pipe(Effect.orElseSucceed(() => ""))
    const requestId = responseHeaders.match(/^x-github-request-id:\s*(.+)$/im)?.[1]?.trim() ?? "unavailable"
    if (metrics.http_code >= 400) {
      return yield* new UploadHttpError({ status: metrics.http_code, message: `POST ${url}: HTTP ${metrics.http_code}; GitHub request ${requestId}; ${body.slice(0, 2_000)}` })
    }
    if (code !== 0 || metrics.http_code !== 201 || metrics.size_upload !== file.bytes) {
      return yield* new UploadTransportError({ message: `POST ${url}: curl exit ${code}; HTTP ${metrics.http_code}; sent ${metrics.size_upload}/${file.bytes} bytes in ${metrics.time_total}s; GitHub request ${requestId}` })
    }
  })).pipe(
    Effect.mapError(error => error instanceof UploadHttpError || error instanceof UploadTransportError ? error : new UploadTransportError({ message: String(error) })),
    Effect.provide(NodeContext.layer),
  ),
})

/** Reconcile against the asset endpoint: release.assets omits incomplete uploads. */
export const uploadReleaseAssets = (options: {
  readonly repository: string
  readonly token: string
  readonly releaseId: number
  readonly tag: string
  readonly sourceCommit: string
  readonly uploadUrl: string
  readonly files: readonly UploadFile[]
}) => Effect.gen(function* () {
  const transport = yield* ReleaseUploadTransport
  const headers = {
    accept: "application/vnd.github+json",
    authorization: `Bearer ${options.token}`,
    "x-github-api-version": "2022-11-28",
  }
  const retry = Schedule.exponential("2 seconds").pipe(Schedule.intersect(Schedule.recurs(3)))
  const retryable = (error: unknown) =>
    error instanceof UploadTransportError ||
    (error instanceof UploadHttpError && (error.status >= 500 || error.status === 422))

  const request = (url: string, init: RequestInit = {}, timeoutMs = 60_000) => Effect.gen(function* () {
    const result = yield* Effect.tryPromise({
      try: async () => {
        const response = await fetch(url, {
          ...init,
          headers: { ...headers, ...init.headers },
          signal: AbortSignal.timeout(timeoutMs),
        })
        return {
          ok: response.ok,
          status: response.status,
          requestId: response.headers.get("x-github-request-id") ?? "unavailable",
          body: await response.text(),
        }
      },
      catch: error => new UploadTransportError({ message: `${init.method ?? "GET"} ${url}: ${String(error)}` }),
    })
    if (!result.ok) {
      return yield* new UploadHttpError({
        status: result.status,
        message: `${init.method ?? "GET"} ${url}: HTTP ${result.status}; GitHub request ${result.requestId}; ${result.body.slice(0, 2_000)}`,
      })
    }
    return result.body
  })
  const api = `https://api.github.com/repos/${options.repository}`
  const draft = yield* request(`${api}/releases/${options.releaseId}`).pipe(
    Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Draft))),
  )
  if (!draft.draft || draft.tag_name !== options.tag || draft.target_commitish !== options.sourceCommit) {
    return yield* new InvalidUploadDraft({ message: "Uploads require the exact private candidate draft" })
  }
  const list = Effect.gen(function* () {
    const assets: Array<typeof Asset.Type> = []
    for (let page = 1; ; page++) {
      const batch = yield* request(`${api}/releases/${options.releaseId}/assets?per_page=100&page=${page}`).pipe(
        Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Array(Asset)))),
      )
      assets.push(...batch)
      if (batch.length < 100) return assets
    }
  })
  const names = new Set(options.files.map(file => file.name))
  const initial = yield* list
  if (initial.some(asset => !names.has(asset.name))) {
    return yield* new InvalidUploadDraft({ message: "Draft contains assets outside the accepted candidate" })
  }
  for (const file of options.files) {
    let attempt = 0
    yield* Effect.gen(function* () {
      attempt++
      let verified = false
      for (const asset of (yield* list).filter(asset => asset.name === file.name)) {
        if (asset.state === "uploaded" && asset.size === file.bytes &&
          Option.getOrNull(asset.digest) === `sha256:${file.sha256}`) {
          verified = true
        } else {
          yield* Console.log(`Removing incomplete or mismatched asset ${file.name} (${asset.state})`)
          yield* request(`${api}/releases/assets/${asset.id}`, { method: "DELETE" })
        }
      }
      if (verified) {
        yield* Console.log(`Verified existing upload ${file.name}`)
        return
      }
      yield* Console.log(`Uploading ${file.name} (${file.bytes} bytes), attempt ${attempt}/4`)
      const started = yield* Effect.clockWith(clock => clock.currentTimeMillis)
      yield* transport.upload(`${options.uploadUrl}?name=${encodeURIComponent(file.name)}`, options.token, file)
      const finished = yield* Effect.clockWith(clock => clock.currentTimeMillis)
      yield* Console.log(`Uploaded ${file.name} in ${finished - started}ms`)
    }).pipe(
      Effect.tapError(error => Console.error(String(error))),
      Effect.retry({ schedule: retry, while: retryable }),
    )
  }
})
