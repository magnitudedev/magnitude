import { redactWorkerOutput } from "./worker-diagnostics"
import { HttpRouter, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import { Effect, Option, Redacted, Schema, Stream } from "effect"
import { Digest, InfrastructureFailure } from "./domain"
import { InvalidResult } from "./work-store"
import { WorkerEvidence } from "./worker-evidence"
import { WorkerInputs } from "./worker-inputs"
import { WorkerInvocation, WorkerReply } from "./worker-protocol"
import { WorkerResults } from "./worker-results"
import { WorkerAccessDenied, WorkerTickets } from "./worker-tickets"

/** A worker credential is deliberately separate from developer/CI authorization. */
export const workerApi = HttpRouter.empty.pipe(
  HttpRouter.put("/v1/worker/evidence/:digest", Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest
    const header = request.headers.authorization
    if (!header?.startsWith("Bearer ")) return yield* new WorkerAccessDenied({})
    const length = request.headers["content-length"]
    if (!length || !/^\d+$/.test(length)) return yield* new InvalidResult({ message: "Evidence requires an exact Content-Length" })
    const digest = yield* Schema.decodeUnknown(Digest)((yield* HttpRouter.params).digest)
    yield* (yield* WorkerEvidence).upload(Redacted.make(header.slice(7)), digest, Number(length), (Number(length) === 0 ? Stream.empty : request.stream).pipe(
      Stream.mapError(() => new InfrastructureFailure({ operation: "worker-evidence", message: "Evidence request interrupted" }))))
    return HttpServerResponse.empty({ status: 204, headers: { "cache-control": "no-store" } })
  })),
  HttpRouter.post("/v1/worker/result", Effect.gen(function* () {
    const header = (yield* HttpServerRequest.HttpServerRequest).headers.authorization
    if (!header?.startsWith("Bearer ")) return yield* new WorkerAccessDenied({})
    const token = Redacted.make(header.slice(7))
    yield* (yield* WorkerTickets).authorize(token)
    const reply = yield* HttpServerRequest.schemaBodyJson(WorkerReply).pipe(HttpServerRequest.withMaxBodySize(Option.some(16 * 1024 * 1024)))
    yield* (yield* WorkerResults).submit(token, reply)
    return HttpServerResponse.empty({ status: 204, headers: { "cache-control": "no-store" } })
  })),
  HttpRouter.get("/v1/worker/objects/:digest", Effect.gen(function* () {
    const header = (yield* HttpServerRequest.HttpServerRequest).headers.authorization
    if (!header?.startsWith("Bearer ")) return yield* new WorkerAccessDenied({})
    const digest = yield* Schema.decodeUnknown(Digest)((yield* HttpRouter.params).digest)
    const content = yield* (yield* WorkerInputs).read(Redacted.make(header.slice(7)), digest)
    return HttpServerResponse.stream(content, { contentType: "application/octet-stream", headers: { "cache-control": "no-store" } })
  })),
  HttpRouter.get("/v1/worker/assignment", Effect.gen(function* () {
    const request = yield* HttpServerRequest.HttpServerRequest
    const header = request.headers.authorization
    if (!header?.startsWith("Bearer ")) return yield* new WorkerAccessDenied({})
    const invocation = yield* (yield* WorkerTickets).authorize(Redacted.make(header.slice(7)))
    return (yield* HttpServerResponse.schemaJson(WorkerInvocation)(invocation)).pipe(HttpServerResponse.setHeader("cache-control", "no-store"))
  })),
  HttpRouter.catchAll(error => Effect.gen(function* () {
    if (error._tag === "InfrastructureFailure") {
      const request = yield* HttpServerRequest.HttpServerRequest
      const token = request.headers.authorization?.replace(/^Bearer /, "")
      const detail = redactWorkerOutput(token ? error.message.replaceAll(token, "[REDACTED]") : error.message).slice(0, 2048)
      yield* Effect.logError("Worker API infrastructure failure").pipe(Effect.annotateLogs({
        operation: error.operation, method: request.method, path: request.url.split("?")[0]!, detail,
      }))
    }
    return HttpServerResponse.unsafeJson({ error: ["WorkerAccessDenied", "InvalidResult", "ParseError", "RequestError"].includes(error._tag) ? error._tag : "WorkerUnavailable" },
      { status: error._tag === "WorkerAccessDenied" ? 401 : error._tag === "InvalidResult" ? 409 : error._tag === "ParseError" || error._tag === "RequestError" ? 400 : 500, headers: { "cache-control": "no-store" } })
  })),
)
