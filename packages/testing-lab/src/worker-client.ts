import { FetchHttpClient, HttpClient, HttpClientRequest } from "@effect/platform"
import { Context, Effect, Layer, Redacted, Schema, Stream } from "effect"
import { verifiedArtifactContent } from "./artifact-store"
import { Digest, InfrastructureFailure } from "./domain"
import { WorkerInvocation, WorkerReply } from "./worker-protocol"

export class WorkerApiError extends Schema.TaggedError<WorkerApiError>()("WorkerApiError", { status: Schema.Int, message: Schema.String }) {}
export interface WorkerClient {
  readonly assignment: Effect.Effect<typeof WorkerInvocation.Type, WorkerApiError>
  readonly download: (digest: Digest) => Stream.Stream<Uint8Array, InfrastructureFailure>
  readonly upload: (digest: Digest, bytes: number, content: Stream.Stream<Uint8Array, InfrastructureFailure>) => Effect.Effect<void, WorkerApiError>
  readonly submit: (reply: typeof WorkerReply.Type) => Effect.Effect<void, WorkerApiError>
}
export const WorkerClient = Context.GenericTag<WorkerClient>("@magnitudedev/testing-lab/WorkerClient")
const failure = (message: string, status = 0) => new WorkerApiError({ status, message })

/** No redirect or automatic mutation retry: a transport failure must never repeat native test execution. */
export const workerClientLayer = (origin: string, token: Redacted.Redacted<string>) => Layer.effect(WorkerClient, Effect.gen(function* () {
  const url = yield* Effect.try({ try: () => new URL(origin), catch: () => failure("Invalid worker coordinator URL") })
  if (url.username || url.password || url.pathname !== "/" || url.search || url.hash ||
    (url.protocol !== "https:" && !(url.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)))) return yield* failure("Worker coordinator requires an HTTPS origin or loopback HTTP")
  const http = yield* HttpClient.HttpClient
  const headers = { authorization: `Bearer ${Redacted.value(token)}` }
  const send = (request: HttpClientRequest.HttpClientRequest, timeoutMs = 30_000) => http.execute(request).pipe(
    Effect.provideService(FetchHttpClient.RequestInit, { redirect: "manual" }),
    Effect.mapError(() => failure("Worker coordinator request failed")),
    Effect.timeoutFail({ duration: timeoutMs, onTimeout: () => failure("Worker coordinator request timed out") }),
    Effect.flatMap(response => response.status >= 200 && response.status < 300 ? Effect.succeed(response) : Effect.fail(failure(`Worker coordinator returned HTTP ${response.status}`, response.status))),
  )
  return {
    assignment: Effect.gen(function* () {
      const response = yield* send(HttpClientRequest.get(`${url.origin}/v1/worker/assignment`, { headers }))
      let size = 0
      const chunks = yield* response.stream.pipe(Stream.mapError(() => failure("Assignment response interrupted")), Stream.tap(chunk => Effect.gen(function* () {
        size += chunk.byteLength
        if (size > 16 * 1024 * 1024) return yield* failure("Assignment exceeds 16 MiB")
      })), Stream.runCollect)
      return yield* Schema.decodeUnknown(Schema.parseJson(WorkerInvocation))(Buffer.concat(Array.from(chunks)).toString("utf8")).pipe(Effect.mapError(() => failure("Malformed worker assignment")))
    }).pipe(Effect.timeoutFail({ duration: "1 minute", onTimeout: () => failure("Assignment response timed out") })),
    download: digest => verifiedArtifactContent(digest, Stream.unwrap(send(HttpClientRequest.get(`${url.origin}/v1/worker/objects/${digest}`, { headers })).pipe(
      Effect.map(response => response.stream.pipe(Stream.mapError(() => new InfrastructureFailure({ operation: "worker-download", message: "Input download interrupted" })))),
      Effect.mapError(error => new InfrastructureFailure({ operation: "worker-download", message: error.message })))), 4 * 1024 ** 3).pipe(
        Stream.timeoutFail(() => new InfrastructureFailure({ operation: "worker-download", message: "Input download stalled" }), "2 minutes")),
    upload: (digest, bytes, content) => Effect.gen(function* () {
      if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > 4 * 1024 ** 3) return yield* failure("Invalid worker object length")
      yield* send(HttpClientRequest.put(`${url.origin}/v1/worker/evidence/${digest}`, { headers }).pipe(
        HttpClientRequest.bodyStream(content, { contentType: "application/octet-stream", contentLength: bytes }), HttpClientRequest.setHeader("content-length", String(bytes))), 10 * 60_000)
    }),
    submit: reply => Effect.gen(function* () {
      const json = yield* Schema.encode(Schema.parseJson(WorkerReply))(reply).pipe(Effect.mapError(() => failure("Invalid worker reply")))
      if (Buffer.byteLength(json) > 16 * 1024 * 1024) return yield* failure("Worker reply exceeds 16 MiB")
      yield* send(HttpClientRequest.post(`${url.origin}/v1/worker/result`, { headers }).pipe(HttpClientRequest.bodyText(json, "application/json")))
    }),
  } satisfies WorkerClient
}))
