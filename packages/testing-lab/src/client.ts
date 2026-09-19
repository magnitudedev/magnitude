import { HttpClient, HttpClientRequest } from "@effect/platform"
import { Context, Effect, Layer, Option, Redacted, Schema, Stream } from "effect"
import { Digest, InfrastructureFailure, Input, ObjectBatch, Principal, RunId, RunPlan, RunRequest, RunResult, Target } from "./domain"
import { RunRecord } from "./run-store"

export class LabApiError extends Schema.TaggedError<LabApiError>()("LabApiError", {
  status: Schema.Int, message: Schema.String,
}) {}
export interface LabClient {
  readonly missing: (digests: readonly Digest[]) => Effect.Effect<readonly Digest[], LabApiError>
  readonly identity: () => Effect.Effect<Principal, LabApiError>
  readonly upload: (digest: Digest, bytes: Stream.Stream<Uint8Array, InfrastructureFailure>) => Effect.Effect<void, LabApiError>
  readonly registerInput: (input: typeof Input.Type) => Effect.Effect<void, LabApiError>
  readonly targets: () => Effect.Effect<ReadonlyArray<Target>, LabApiError>
  readonly plan: (request: RunRequest) => Effect.Effect<RunPlan, LabApiError>
  readonly submit: (request: RunRequest) => Effect.Effect<RunRecord, LabApiError>
  readonly get: (id: RunId) => Effect.Effect<RunRecord, LabApiError>
  readonly cancel: (id: RunId) => Effect.Effect<RunRecord, LabApiError>
  readonly result: (id: RunId) => Effect.Effect<Option.Option<RunResult>, LabApiError>
}
export const LabClient = Context.GenericTag<LabClient>("@magnitudedev/testing-lab/LabClient")
export const labClientLayer = (origin: string, token: Effect.Effect<Redacted.Redacted<string>, LabApiError>) => Layer.effect(LabClient, Effect.gen(function* () {
  const url = yield* Effect.try({ try: () => new URL(origin), catch: () => new LabApiError({ status: 0, message: "Invalid coordinator URL" }) })
  if (url.username || url.password || url.pathname !== "/" || url.search || url.hash ||
    (url.protocol !== "https:" && !(url.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)))) {
    return yield* new LabApiError({ status: 0, message: "Coordinator must be an HTTPS origin or a loopback HTTP origin" })
  }
  const http = yield* HttpClient.HttpClient
  const send = (path: string, body: Option.Option<RunRequest>, post = false) => Effect.gen(function* () {
    let request = (post ? HttpClientRequest.post : HttpClientRequest.get)(`${url.origin}${path}`,
      { headers: { authorization: `Bearer ${Redacted.value(yield* token)}` } })
    if (Option.isSome(body)) {
      const json = yield* Schema.encode(Schema.parseJson(RunRequest))(body.value).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "Invalid run request" })))
      request = HttpClientRequest.bodyText(request, json, "application/json")
    }
    const response = yield* http.execute(request).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "Coordinator request failed" })))
    if (response.status < 200 || response.status >= 300) return yield* new LabApiError({ status: response.status, message: `Coordinator returned HTTP ${response.status}` })
    return response
  }).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new LabApiError({ status: 0, message: "Coordinator request timed out" }) }))
  const read = <A, I>(schema: Schema.Schema<A, I>, path: string, body: Option.Option<RunRequest> = Option.none(), post = false) => Effect.gen(function* () {
    const response = yield* send(path, body, post)
    const json = yield* response.json.pipe(Effect.mapError(() => new LabApiError({ status: response.status, message: "Coordinator returned invalid JSON" })))
    return yield* Schema.decodeUnknown(schema)(json).pipe(Effect.mapError(() => new LabApiError({ status: response.status, message: "Coordinator response violates the lab protocol" })))
  }).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new LabApiError({ status: 0, message: "Coordinator response body timed out" }) }))
  return {
    missing: digests => Effect.gen(function* () {
      const json = yield* Schema.encode(Schema.parseJson(ObjectBatch))(digests).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "Object batches allow at most 1000 digests" })))
      const response = yield* http.execute(HttpClientRequest.post(`${url.origin}/v1/objects/missing`, { headers: { authorization: `Bearer ${Redacted.value(yield* token)}` } }).pipe(HttpClientRequest.bodyText(json, "application/json"))).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "Object availability request failed" })))
      if (response.status !== 200) return yield* new LabApiError({ status: response.status, message: "Object availability request rejected" })
      return yield* response.json.pipe(Effect.flatMap(Schema.decodeUnknown(ObjectBatch)), Effect.mapError(() => new LabApiError({ status: 0, message: "Malformed object availability response" })))
    }).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new LabApiError({ status: 0, message: "Object availability request timed out" }) })),
    upload: (digest, bytes) => Effect.gen(function* () {
      const request = HttpClientRequest.put(`${url.origin}/v1/objects/${digest}`, { headers: { authorization: `Bearer ${Redacted.value(yield* token)}` } }).pipe(
        HttpClientRequest.bodyStream(bytes, { contentType: "application/octet-stream" }),
      )
      const response = yield* http.execute(request).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "Object upload failed" })))
      if (response.status !== 204) return yield* new LabApiError({ status: response.status, message: "Object upload rejected" })
    }).pipe(Effect.timeoutFail({ duration: "10 minutes", onTimeout: () => new LabApiError({ status: 0, message: "Object upload timed out" }) })),
    registerInput: input => Effect.gen(function* () {
      const json = yield* Schema.encode(Schema.parseJson(Input))(input).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "Invalid input reference" })))
      const request = HttpClientRequest.post(`${url.origin}/v1/inputs`, { headers: { authorization: `Bearer ${Redacted.value(yield* token)}` } }).pipe(HttpClientRequest.bodyText(json, "application/json"))
      const response = yield* http.execute(request).pipe(Effect.mapError(() => new LabApiError({ status: 0, message: "Input registration failed" })))
      if (response.status !== 204) return yield* new LabApiError({ status: response.status, message: "Input registration rejected" })
    }).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new LabApiError({ status: 0, message: "Input registration timed out" }) })),
    identity: () => read(Principal, "/v1/me"),
    targets: () => read(Schema.Array(Target), "/v1/targets"),
    plan: request => read(RunPlan, "/v1/runs/plan", Option.some(request), true),
    submit: request => read(RunRecord, "/v1/runs", Option.some(request), true),
    get: id => read(RunRecord, `/v1/runs/${encodeURIComponent(id)}`),
    cancel: id => read(RunRecord, `/v1/runs/${encodeURIComponent(id)}/cancel`, Option.none(), true),
    result: id => Effect.gen(function* () {
      const response = yield* send(`/v1/runs/${encodeURIComponent(id)}/results`, Option.none())
      if (response.status === 202) return Option.none()
      const json = yield* response.json.pipe(Effect.mapError(() => new LabApiError({ status: response.status, message: "Coordinator returned invalid JSON" })))
      return Option.some(yield* Schema.decodeUnknown(RunResult)(json).pipe(Effect.mapError(() => new LabApiError({ status: response.status, message: "Malformed run result" }))))
    }).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new LabApiError({ status: 0, message: "Result response timed out" }) })),
  } satisfies LabClient
}))
