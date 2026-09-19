import { HttpRouter, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import { Effect, Redacted, Schema } from "effect"
import { Digest } from "./domain"
import { WorkerInputs } from "./worker-inputs"
import { WorkerInvocation } from "./worker-protocol"
import { WorkerAccessDenied, WorkerTickets } from "./worker-tickets"

/** A worker credential is deliberately separate from developer/CI authorization. */
export const workerApi = HttpRouter.empty.pipe(
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
  HttpRouter.catchAll(error => Effect.succeed(HttpServerResponse.unsafeJson({ error: error._tag === "WorkerAccessDenied" ? "WorkerAccessDenied" : "WorkerUnavailable" },
    { status: error._tag === "WorkerAccessDenied" ? 401 : 500, headers: { "cache-control": "no-store" } }))),
)
