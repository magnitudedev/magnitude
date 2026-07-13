/**
 * Remote daemon spawner — browser-safe HTTP delegate.
 *
 * `makeRemoteDaemonSpawner` is an `Effect` that captures `HttpClient` at
 * construction time, returning a sealed `DaemonSpawner` whose methods require
 * only `never`. This keeps the `DaemonSpawner` interface clean while the layer
 * has the real requirements.
 *
 * Both `discover` and `spawn` delegate to a proxy server's HTTP endpoints
 * (`GET /discover`, `POST /spawn`). The proxy server (a thin Bun process
 * started by `bun web`) has the actual spawn capability — it reads
 * registration files, health-checks, and spawns processes. The browser
 * cannot do any of that, so it delegates.
 *
 * The browser uses the same `DaemonSpawner` abstraction as everyone else —
 * the spawner just happens to be remote. This is the single abstraction point
 * that keeps the browser path aligned with local Bun/Node spawners.
 *
 * Browser-safe: only depends on `HttpClient` (fetch-based). Zero Node imports.
 */
import { Effect, Option, Schema } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { DaemonSpawnFailed } from "./errors"
import type { DaemonSpawner } from "./daemon-spawner"

// ─── Response schemas ────────────────────────────────────────────────────────

const DiscoverResponse = Schema.Struct({
  url: Schema.Union(Schema.String, Schema.Null),
})

const SpawnResponse = Schema.Struct({
  url: Schema.String,
})

const ErrorResponse = Schema.Struct({
  error: Schema.String,
})

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Extract error message from a failed HTTP response body. */
const extractErrorMessage = (
  response: HttpClientResponse.HttpClientResponse
): Effect.Effect<string> =>
  response.json.pipe(
    Effect.catchAll(() => Effect.succeed(null)),
    Effect.flatMap((json) =>
      Schema.decodeUnknown(ErrorResponse)(json).pipe(
        Effect.map((body) => body.error),
        Effect.catchAll(() => Effect.succeed(`HTTP ${response.status}`)),
      )
    ),
  )

// ─── Public factory ──────────────────────────────────────────────────────────

/**
 * Creates a remote `DaemonSpawner` that delegates `discover` and `spawn` to
 * a proxy server via HTTP fetch.
 *
 * The returned spawner's methods require only `never` — `HttpClient` is
 * captured at construction time and sealed inside.
 *
 * @param proxyUrl — base URL of the proxy server (e.g. `http://127.0.0.1:53108`)
 */
export const makeRemoteDaemonSpawner = (
  proxyUrl: string
): Effect.Effect<DaemonSpawner, never, HttpClient.HttpClient> =>
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient

    return {
      // GET {proxyUrl}/discover → { url: string | null }
      // Best-effort: any error (network failure, timeout, parse error,
      // non-200) returns None. The caller will then try spawn.
      discover: () =>
        client.execute(
          HttpClientRequest.get(`${proxyUrl}/discover`)
        ).pipe(
          Effect.timeout("2 seconds"),
          Effect.flatMap((response) => response.json),
          Effect.flatMap((json) => Schema.decodeUnknown(DiscoverResponse)(json)),
          Effect.map((body) =>
            body.url === null
              ? Option.none<string>()
              : Option.some(body.url)
          ),
          Effect.catchAll(() => Effect.succeed(Option.none<string>())),
        ),

      // POST {proxyUrl}/spawn with { command } → { url: string }
      // Hard failure on any error — spawn is the fallback path, if it fails
      // there's nothing else to try.
      spawn: (command) =>
        Effect.gen(function* () {
          const req = yield* HttpClientRequest.post(`${proxyUrl}/spawn`).pipe(
            HttpClientRequest.bodyJson({ command }),
            Effect.mapError((cause) => new DaemonSpawnFailed({ reason: `Failed to build request: ${String(cause)}` })),
          )
          const response = yield* client.execute(req).pipe(
            Effect.timeout("30 seconds"),
            Effect.mapError((cause) =>
              new DaemonSpawnFailed({ reason: `Proxy request timed out: ${String(cause)}` })
            ),
          )

          if (response.status < 200 || response.status >= 300) {
            const message = yield* extractErrorMessage(response)
            return yield* new DaemonSpawnFailed({ reason: `Proxy returned error: ${message}` })
          }

          const json = yield* response.json.pipe(
            Effect.mapError((cause) =>
              new DaemonSpawnFailed({ reason: `Failed to parse proxy response: ${String(cause)}` })
            ),
          )

          const parsed = yield* Schema.decodeUnknown(SpawnResponse)(json).pipe(
            Effect.mapError((cause) =>
              new DaemonSpawnFailed({ reason: `Invalid proxy response: ${String(cause)}` })
            ),
          )

          return parsed.url
        }),
    } satisfies DaemonSpawner
  })
