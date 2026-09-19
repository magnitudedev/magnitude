import { HttpRouter, HttpServerRequest, HttpServerResponse } from "@effect/platform"
import { Context, Effect, Layer, Option, Redacted, Schema, Stream } from "effect"
import { timingSafeEqual } from "node:crypto"
import { Digest, InfrastructureFailure, Input, ObjectBatch, Principal, RunId, RunPlan, RunRequest, RunResult, Target, runInputs } from "./domain"
import { planRun, targets } from "./catalog"
import { RunRecord, RunStore } from "./run-store"
import { InputRegistry } from "./inputs"
import { ArtifactStore, verifiedArtifactContent } from "./artifact-store"

export class Unauthorized extends Schema.TaggedError<Unauthorized>()("Unauthorized", {}) {}
export class Forbidden extends Schema.TaggedError<Forbidden>()("Forbidden", {}) {}
export interface Authenticator {
  readonly authenticate: (authorization: string | undefined) => Effect.Effect<Principal, Unauthorized>
}
export const Authenticator = Context.GenericTag<Authenticator>("@magnitudedev/testing-lab/Authenticator")
/** Local/private deployment credentials map to server-owned identities, never payload-provided privilege. */
export const bearerAuthenticator = (credentials: ReadonlyArray<{ readonly token: Redacted.Redacted<string>; readonly principal: Principal }>) => Layer.succeed(Authenticator, {
  authenticate: header => Effect.gen(function* () {
    if (!header?.startsWith("Bearer ")) return yield* new Unauthorized({})
    const supplied = Buffer.from(header.slice(7))
    for (const credential of credentials) {
      const expected = Buffer.from(Redacted.value(credential.token))
      if (expected.length >= 32 && supplied.length === expected.length && timingSafeEqual(supplied, expected)) return credential.principal
    }
    return yield* new Unauthorized({})
  }),
})
const principal = Effect.gen(function* () {
  const request = yield* HttpServerRequest.HttpServerRequest
  return yield* (yield* Authenticator).authenticate(request.headers.authorization)
})
const admittedRequest = Effect.gen(function* () {
  const identity = yield* principal
  const request = yield* HttpServerRequest.schemaBodyJson(RunRequest)
  if (request.owner !== identity.owner || request.trust !== identity.trust) return yield* new Forbidden({})
  return request
})
const ownedRun = Effect.gen(function* () {
  const identity = yield* principal
  const params = yield* HttpRouter.params
  const id = yield* Schema.decodeUnknown(RunId)(params.id)
  const store = yield* RunStore
  const run = yield* store.get(id)
  if (run.state.plan.request.owner !== identity.owner) return yield* new Forbidden({})
  return { id, store, run }
})
const errorResponse = (error: { readonly _tag: string }) => {
  const status = error._tag === "Unauthorized" ? 401 : error._tag === "Forbidden" || error._tag === "InputDenied" ? 403 : error._tag === "RunNotFound" ? 404
    : error._tag === "AdmissionRejected" ? 409 : error._tag === "InvalidInput" || error._tag === "ParseError" ? 400 : 500
  return HttpServerResponse.unsafeJson({ error: error._tag }, { status })
}
export const api = HttpRouter.empty.pipe(
  HttpRouter.put("/v1/objects/:digest", Effect.gen(function* () {
    const identity = yield* principal
    const digest = yield* Schema.decodeUnknown(Digest)((yield* HttpRouter.params).digest)
    const request = yield* HttpServerRequest.HttpServerRequest
    const stream = request.stream.pipe(Stream.mapError(() => new InfrastructureFailure({ operation: "upload", message: "Object upload interrupted" })))
    yield* (yield* InputRegistry).upload(identity.owner, digest, stream)
    return HttpServerResponse.empty({ status: 204 })
  })),
  HttpRouter.get("/v1/objects/:digest", Effect.gen(function* () {
    const identity = yield* principal
    const digest = yield* Schema.decodeUnknown(Digest)((yield* HttpRouter.params).digest)
    return HttpServerResponse.stream(yield* (yield* InputRegistry).read(identity.owner, digest), { contentType: "application/octet-stream" })
  })),
  HttpRouter.post("/v1/inputs", Effect.gen(function* () {
    const identity = yield* principal
    const input = yield* HttpServerRequest.schemaBodyJson(Input)
    yield* (yield* InputRegistry).register(identity.owner, input)
    return HttpServerResponse.empty({ status: 204 })
  })),
  HttpRouter.post("/v1/objects/missing", Effect.gen(function* () {
    const identity = yield* principal
    const digests = yield* HttpServerRequest.schemaBodyJson(ObjectBatch)
    return yield* HttpServerResponse.schemaJson(ObjectBatch)(yield* (yield* InputRegistry).missing(identity.owner, digests))
  })),
  HttpRouter.get("/v1/me", principal.pipe(Effect.flatMap(HttpServerResponse.schemaJson(Principal)))),
  HttpRouter.get("/v1/targets", principal.pipe(Effect.zipRight(HttpServerResponse.schemaJson(Schema.Array(Target))(targets)))),
  HttpRouter.post("/v1/runs/plan", Effect.gen(function* () {
    const plan = yield* planRun(yield* admittedRequest)
    return yield* HttpServerResponse.schemaJson(RunPlan)(plan)
  })),
  HttpRouter.post("/v1/runs", Effect.gen(function* () {
    const plan = yield* planRun(yield* admittedRequest)
    const inputs = yield* InputRegistry
    yield* Effect.forEach(runInputs(plan.request), input => inputs.require(plan.request.owner, input), { discard: true })
    const run = yield* (yield* RunStore).submit(plan)
    return yield* HttpServerResponse.schemaJson(RunRecord)(run, { status: 201 })
  })),
  HttpRouter.get("/v1/runs/:id", ownedRun.pipe(Effect.flatMap(({ run }) => HttpServerResponse.schemaJson(RunRecord)(run)))),
  HttpRouter.get("/v1/runs/:id/results", Effect.gen(function* () {
    const { id, store } = yield* ownedRun
    const result = yield* store.result(id)
    return Option.isSome(result) ? yield* HttpServerResponse.schemaJson(RunResult)(result.value)
      : HttpServerResponse.unsafeJson({ status: "pending" }, { status: 202 })
  })),
  HttpRouter.get("/v1/runs/:id/evidence/:digest", Effect.gen(function* () {
    const { id, store } = yield* ownedRun
    const digest = yield* Schema.decodeUnknown(Digest)((yield* HttpRouter.params).digest)
    const result = yield* store.result(id)
    const evidence = Option.isSome(result) ? result.value.cases.flatMap(test => test.evidence).find(item => item.sha256 === digest) : undefined
    if (!evidence) return HttpServerResponse.empty({ status: 404 })
    return HttpServerResponse.stream(verifiedArtifactContent(digest, (yield* ArtifactStore).get(digest), evidence.bytes), {
      contentType: "application/octet-stream",
    })
  })),
  HttpRouter.post("/v1/runs/:id/cancel", Effect.gen(function* () {
    const { id, store } = yield* ownedRun
    return yield* HttpServerResponse.schemaJson(RunRecord)(yield* store.cancel(id))
  })),
  HttpRouter.catchAll(error => Effect.succeed(errorResponse(error))),
)
