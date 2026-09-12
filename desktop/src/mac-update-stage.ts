import { HttpServer, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import { NodeHttpServer } from "@effect/platform-node"
import { Context, Effect, Schema } from "effect"
import type { AutoUpdater } from "electron"
import { randomUUID } from "node:crypto"
import { createServer } from "node:http"

export class MacUpdateStageFailed extends Schema.TaggedError<MacUpdateStageFailed>()("MacUpdateStageFailed", {
  message: Schema.String,
}) {}

export interface NativeMacUpdate {
  readonly stage: (feed: { readonly url: string; readonly authorization: string }) => Effect.Effect<void, MacUpdateStageFailed>
}
export const NativeMacUpdate = Context.GenericTag<NativeMacUpdate>("desktop/NativeMacUpdate")

/** Electron owns its native staging occurrence; this adapter observes its terminal event. */
export const nativeMacUpdate = (updater: AutoUpdater): NativeMacUpdate => ({
  stage: feed => Effect.async<void, MacUpdateStageFailed>(resume => {
    const downloaded = () => { cleanup(); resume(Effect.void) }
    const failed = (error: Error) => { cleanup(); resume(new MacUpdateStageFailed({ message: error.message })) }
    const cleanup = () => {
      updater.removeListener("update-downloaded", downloaded)
      updater.removeListener("error", failed)
    }
    updater.once("update-downloaded", downloaded)
    updater.once("error", failed)
    try {
      updater.setFeedURL({ url: feed.url, headers: { Authorization: feed.authorization } })
      updater.checkForUpdates()
    } catch (error) {
      cleanup()
      resume(new MacUpdateStageFailed({ message: error instanceof Error ? error.message : "The native updater could not start." }))
    }
    return Effect.sync(cleanup)
  }),
})

const Feed = Schema.Struct({ url: Schema.String })

/** The caller retains the verified archive for this scope; no directory is exposed over HTTP. */
export const stageMacUpdateArchive = (archive: string) => Effect.scoped(Effect.gen(function* () {
  const native = yield* NativeMacUpdate
  const server = yield* HttpServer.HttpServer
  if (server.address._tag !== "TcpAddress") return yield* new MacUpdateStageFailed({ message: "The local update endpoint could not start." })
  const origin = `http://127.0.0.1:${server.address.port}`
  const authorization = `Bearer ${randomUUID()}`
  const archiveRoute = `/${randomUUID()}.zip`
  yield* HttpServer.serveEffect(Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest
    if (request.method !== "GET") return HttpServerResponse.empty({ status: 405 })
    if (request.url === "/feed" && request.headers.authorization === authorization) {
      return yield* HttpServerResponse.schemaJson(Feed)({ url: origin + archiveRoute })
    }
    if (request.url === archiveRoute) {
      return yield* HttpServerResponse.file(archive, { headers: { "content-type": "application/zip", "cache-control": "no-store" } })
    }
    return HttpServerResponse.empty({ status: 404 })
  }).pipe(Effect.catchAll(() => Effect.succeed(HttpServerResponse.empty({ status: 500 })))))
  yield* native.stage({ url: origin + "/feed", authorization }).pipe(Effect.timeoutFail({
    duration: "3 minutes", onTimeout: () => new MacUpdateStageFailed({ message: "The native updater did not finish staging the downloaded application." }),
  }))
})).pipe(
  Effect.provide(NodeHttpServer.layer(createServer, { host: "127.0.0.1", port: 0 })),
  Effect.catchTag("ServeError", () => new MacUpdateStageFailed({ message: "The local update endpoint could not start." })),
)
