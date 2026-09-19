import { Effect, Ref, Runtime, Schema } from "effect"
import { randomUUID } from "node:crypto"
import { InfrastructureFailure } from "./domain"

export const ExecutionTraceId = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{32}$/),
  Schema.filter(value => value !== "0".repeat(32)), Schema.brand("ExecutionTraceId"))
const PositiveInteger = Schema.Int.pipe(Schema.positive(), Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER))
const NativeId = Schema.String.pipe(Schema.pattern(/^[1-9][0-9]{0,19}$/),
  Schema.filter(value => BigInt(value) <= 18446744073709551615n), Schema.brand("NativeExecutionId"))
export const NativeModelAllocation = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("host"), model_bytes: PositiveInteger }),
  Schema.Struct({ kind: Schema.Literal("device"), backend: Schema.NonEmptyString.pipe(Schema.maxLength(256)),
    physical_id: Schema.NullOr(Schema.NonEmptyString.pipe(Schema.maxLength(256))), native_index: Schema.Int.pipe(Schema.nonNegative(), Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER)), model_bytes: PositiveInteger }),
)
export const NativeExecution = Schema.Struct({ traceId: ExecutionTraceId, model: Schema.NonEmptyString.pipe(Schema.maxLength(1024)),
  workerPid: PositiveInteger, workerGeneration: NativeId, requestId: NativeId, allocations: Schema.Array(NativeModelAllocation) })
export type NativeExecution = typeof NativeExecution.Type
const Attribute = Schema.Struct({ key: Schema.String, value: Schema.Unknown })
const Attributes = Schema.Array(Attribute)
const Envelope = Schema.Struct({ resourceLogs: Schema.Array(Schema.Struct({
  resource: Schema.Struct({ attributes: Attributes }), scopeLogs: Schema.Array(Schema.Struct({ logRecords: Schema.Array(Schema.Unknown) })),
})) })
const Log = Schema.Struct({ attributes: Attributes })
const TraceLog = Schema.Struct({ traceId: ExecutionTraceId })
const StringValue = Schema.Struct({ stringValue: Schema.String })
const IntegerValue = Schema.Struct({ intValue: Schema.Union(Schema.String, PositiveInteger) })
const failure = (message: string) => new InfrastructureFailure({ operation: "execution-telemetry", message })
const attributeMap = (attributes: typeof Attributes.Type) => Effect.gen(function* () {
  const values = new Map<string, unknown>()
  for (const attribute of attributes) {
    if (values.has(attribute.key)) return yield* failure("Duplicate telemetry attribute")
    values.set(attribute.key, attribute.value)
  }
  return values
})

/** Extract only the reviewed completion fields. Never retain arbitrary log bodies or attributes. */
export const decodeNativeExecutions = (input: unknown) => Effect.gen(function* () {
  const envelope = yield* Schema.decodeUnknown(Envelope)(input)
  const records: NativeExecution[] = []
  for (const resource of envelope.resourceLogs) {
    const attributes = yield* attributeMap(resource.resource.attributes)
    const service = yield* Schema.decodeUnknown(StringValue)(attributes.get("service.name"))
    if (service.stringValue !== "magnitude-icn") continue
    for (const scope of resource.scopeLogs) for (const raw of scope.logRecords) {
      const log = yield* Schema.decodeUnknown(Log)(raw)
      const attributes = yield* attributeMap(log.attributes)
      const event = yield* Schema.decodeUnknown(StringValue)(attributes.get("event.name")).pipe(Effect.option)
      if (event._tag === "None" || event.value.stringValue !== "icn.inference.completed") continue
      const trace = yield* Schema.decodeUnknown(TraceLog)(raw)
      const string = (key: string) => Schema.decodeUnknown(StringValue)(attributes.get(key)).pipe(Effect.map(value => value.stringValue))
      const integer = (key: string) => Schema.decodeUnknown(IntegerValue)(attributes.get(key)).pipe(
        Effect.map(value => String(value.intValue)), Effect.flatMap(Schema.decodeUnknown(NativeId)))
      const allocationJson = yield* string("native.target.allocations")
      if (Buffer.byteLength(allocationJson) > 16 * 1024) return yield* failure("Native allocation diagnostic exceeds its byte limit")
      const allocations = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Array(NativeModelAllocation)))(allocationJson)
      records.push(yield* Schema.decodeUnknown(NativeExecution)({ traceId: trace.traceId, model: yield* string("model.id"),
        workerPid: Number(yield* integer("worker.pid")), workerGeneration: yield* integer("worker.generation"),
        requestId: yield* integer("worker.request.id"), allocations }))
    }
  }
  return records
}).pipe(Effect.mapError(() => failure("Invalid native execution telemetry")))

/** Scoped local OTLP/JSON sink. Trace payloads are acknowledged but not retained. */
export const executionTelemetry = () => Effect.gen(function* () {
  const records = yield* Ref.make<readonly NativeExecution[]>([])
  const invalid = yield* Ref.make(false)
  const gate = yield* Effect.makeSemaphore(1)
  const prefix = `/${randomUUID()}`
  const runtime = yield* Effect.runtime<never>()
  const handle = (request: Request) => Effect.gen(function* () {
    const url = new URL(request.url)
    if (request.method !== "POST" || url.search || ![`${prefix}/v1/logs`, `${prefix}/v1/traces`].includes(url.pathname)) return new Response(null, { status: 404 })
    if (!request.headers.get("content-type")?.startsWith("application/json")) return new Response(null, { status: 415 })
    if (url.pathname.endsWith("/v1/traces")) { yield* Effect.tryPromise(() => request.arrayBuffer()); return Response.json({}) }
    const input = yield* Effect.tryPromise(() => request.json())
    const received = yield* decodeNativeExecutions(input)
    return yield* gate.withPermits(1)(Effect.gen(function* () {
      // A complete batch is validated before publication; concurrent export requests cannot lose records.
      const current = yield* Ref.get(records)
      const next = [...current]
      for (const record of received) {
        const previous = next.find(value => value.traceId === record.traceId && value.workerPid === record.workerPid
          && value.workerGeneration === record.workerGeneration && value.requestId === record.requestId)
        if (previous) {
          if (!Schema.equivalence(NativeExecution)(previous, record)) return yield* failure("Conflicting native completion records")
        } else next.push(record)
        if (next.length > 4096) return yield* failure("Native execution retention limit exceeded")
      }
      yield* Ref.set(records, next)
      return Response.json({})
    }))
  }).pipe(Effect.catchAll(() => Ref.set(invalid, true).pipe(Effect.as(new Response(null, { status: 400 })))))
  const server = yield* Effect.acquireRelease(Effect.try({ try: () => Bun.serve({ hostname: "127.0.0.1", port: 0,
    maxRequestBodySize: 1024 * 1024, fetch: request => Runtime.runPromise(runtime)(handle(request)),
  }), catch: () => failure("Could not start native execution collector") }), server => Effect.promise(() => server.stop(true)))
  const observations = Effect.gen(function* () {
    if (yield* Ref.get(invalid)) return yield* failure("Native execution collection contains rejected records")
    return yield* Ref.get(records)
  })
  return { endpoint: `http://127.0.0.1:${server.port}${prefix}`, observations }
})
