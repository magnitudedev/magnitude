import { InfrastructureFailure } from "../src/domain"
import { workerApi } from "../src/worker-api"
import { WorkerEvidence } from "../src/worker-evidence"
import { WorkerInputs } from "../src/worker-inputs"
import { WorkerResults } from "../src/worker-results"
import { WorkerTickets } from "../src/worker-tickets"
import { FetchHttpClient, HttpRouter, HttpServer, HttpServerResponse } from "@effect/platform"
import { BunHttpServer } from "@effect/platform-bun"
import { Context, Effect, Layer, Logger, Redacted, Stream } from "effect"
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

for (const variant of [0, 1, 32768, "failure"] as const) test(`worker evidence crosses the native HTTP boundary: ${variant}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const server = yield* HttpServer.HttpServer
  if (server.address._tag !== "TcpAddress") return yield* Effect.dieMessage("Expected TCP server")
  const size = variant === "failure" ? 1 : variant
  const logs: string[] = []
  const logger = Logger.make(({ message, annotations }) => { logs.push(JSON.stringify({ message, annotations: Array.from(annotations) })) })
  const content = new Uint8Array(size).fill(65)
  let received = -1
  const unused = () => Effect.dieMessage("Unused worker API fixture method")
  yield* server.serve(workerApi.pipe(
    Effect.provide(Logger.replace(Logger.defaultLogger, logger)),
    Effect.provideService(WorkerEvidence, { upload: (_token, digest, bytes, stream) => variant === "failure" ? Effect.fail(new InfrastructureFailure({ operation: "fixture-upload", message: "Disk is full; fixture-token https://example.com/upload?sig=private-capability" })) : Stream.runCollect(stream).pipe(Effect.tap(chunks => Effect.sync(() => {
      const actual = Buffer.concat(Array.from(chunks))
      expect(bytes).toBe(size)
      expect(sha256(actual)).toBe(digest)
      received = actual.byteLength
    })), Effect.asVoid) }),
    Effect.provideService(WorkerInputs, { read: unused }),
    Effect.provideService(WorkerResults, { read: unused, submit: unused }),
    Effect.provideService(WorkerTickets, { issue: unused, revoke: unused, authorize: unused, withAuthority: unused }),
  ))
  const client = Context.get(yield* Layer.build(workerClientLayer(`http://127.0.0.1:${server.address.port}`, Redacted.make("fixture-token"))), WorkerClient)
  const result = yield* client.upload(sha256(content), size, Stream.make(content)).pipe(Effect.either)
  if (variant === "failure") {
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left.status).toBe(500)
      expect(result.left.message).toContain(`PUT /v1/worker/evidence/${sha256(content)}`)
      expect(result.left.message).not.toContain("Disk is full")
    }
    expect(logs).toHaveLength(1)
    expect(logs[0]).toContain("fixture-upload")
    expect(logs[0]).toContain("Disk is full")
    expect(logs[0]).not.toContain("fixture-token")
    expect(logs[0]).not.toContain("private-capability")
  } else {
    expect(result._tag).toBe("Right")
    expect(received).toBe(size)
    expect(logs).toEqual([])
  }
})).pipe(Effect.provide([BunHttpServer.layer({ hostname: "127.0.0.1", port: 0 }), FetchHttpClient.layer]))))
