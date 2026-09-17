import { Console, Effect, Option, Schedule, Schema } from "effect"

class UploadHttpError extends Schema.TaggedError<UploadHttpError>()("UploadHttpError", {
  status: Schema.Number,
  message: Schema.String,
}) {}

class UploadTransportError extends Schema.TaggedError<UploadTransportError>()("UploadTransportError", {
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
  const headers = {
    accept: "application/vnd.github+json",
    authorization: `Bearer ${options.token}`,
    "x-github-api-version": "2022-11-28",
  }
  const retry = Schedule.exponential("2 seconds").pipe(Schedule.intersect(Schedule.recurs(3)))
  const retryable = (error: unknown) =>
    error instanceof UploadTransportError ||
    (error instanceof UploadHttpError && (error.status >= 500 || error.status === 422))

  const request = (url: string, init: RequestInit = {}) => Effect.gen(function* () {
    const result = yield* Effect.tryPromise({
      try: async () => {
        const response = await fetch(url, {
          ...init,
          headers: { ...headers, ...init.headers },
          signal: AbortSignal.timeout(init.method === "POST" ? 30 * 60_000 : 60_000),
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
      yield* request(`${options.uploadUrl}?name=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: Bun.file(file.path),
      })
      const finished = yield* Effect.clockWith(clock => clock.currentTimeMillis)
      yield* Console.log(`Uploaded ${file.name} in ${finished - started}ms`)
    }).pipe(
      Effect.tapError(error => Console.error(String(error))),
      Effect.retry({ schedule: retry, while: retryable }),
    )
  }
})
