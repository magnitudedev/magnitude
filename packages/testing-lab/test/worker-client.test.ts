import { FetchHttpClient, HttpRouter, HttpServer, HttpServerResponse } from "@effect/platform"
import { BunHttpServer } from "@effect/platform-bun"
import { Context, Effect, Layer, Redacted, Stream } from "effect"
import { expect, test } from "vitest"
import { sha256 } from "../src/snapshot"
import { WorkerClient, workerClientLayer } from "../src/worker-client"

test("worker requests reject redirects and corrupt content without retrying", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const server = yield* HttpServer.HttpServer
  if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP server")
  const origin = `http://127.0.0.1:${server.address.port}`
  let assignmentCalls = 0, redirectCalls = 0, downloadCalls = 0
  yield* server.serve(HttpRouter.empty.pipe(
    HttpRouter.get("/v1/worker/assignment", Effect.sync(() => { assignmentCalls++; return HttpServerResponse.redirect(`${origin}/redirect-target`) })),
    HttpRouter.get("/redirect-target", Effect.sync(() => { redirectCalls++; return HttpServerResponse.text("must not follow") })),
    HttpRouter.get("/v1/worker/objects/:digest", Effect.sync(() => { downloadCalls++; return HttpServerResponse.text("corrupt bytes") })),
  ))
  const client = Context.get(yield* Layer.build(workerClientLayer(origin, Redacted.make("fixture-token"))), WorkerClient)
  const result = yield* client.assignment.pipe(Effect.either)
  expect(result._tag === "Left" && result.left.status).toBe(302)
  expect(assignmentCalls).toBe(1)
  expect(redirectCalls).toBe(0)
  expect((yield* client.download(sha256("expected bytes")).pipe(Stream.runDrain, Effect.either))._tag).toBe("Left")
  expect(downloadCalls).toBe(1)
})).pipe(Effect.provide([BunHttpServer.layer({ hostname: "127.0.0.1", port: 0 }), FetchHttpClient.layer]))))

test.each(["http://example.com", "https://user:password@example.com", "https://example.com/path", "https://example.com?token=secret", "file:///tmp/worker"])("rejects unsafe worker origin %s", origin => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const result = yield* Layer.build(workerClientLayer(origin, Redacted.make("fixture-token"))).pipe(Effect.either)
  expect(result._tag).toBe("Left")
})).pipe(Effect.provide(FetchHttpClient.layer))))
