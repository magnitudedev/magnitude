import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Data, Duration, Effect, Option, Redacted, Schema, Stream, pipe } from "effect"
import { formatSchemaIssues, type SchemaIssue } from "../schema-issues"
import {
  LlamaServedModelId as LlamaServedModelIdSchema,
  LlamaInstanceId as LlamaInstanceIdSchema,
  NormalizedLlamaModelPath,
  normalizeLlamaModelPath,
  type LlamaInstanceId,
  type LlamaServedModelId,
} from "./identity"

export const Availability = Schema.Literal("supported", "unsupported", "unknown")
export type Availability = Schema.Schema.Type<typeof Availability>
export const LlamaServedModelStatus = Schema.Literal("unloaded", "loading", "loaded", "sleeping", "downloading", "failed", "unknown")
export type LlamaServedModelStatus = Schema.Schema.Type<typeof LlamaServedModelStatus>
export const LlamaInstanceOwnership = Schema.Literal("managed", "external")
export type LlamaInstanceOwnership = Schema.Schema.Type<typeof LlamaInstanceOwnership>
export const LlamaServerHealth = Schema.Literal("ready", "loading", "unavailable")
export type LlamaServerHealth = Schema.Schema.Type<typeof LlamaServerHealth>
export const LlamaServerMode = Schema.Literal("router", "single-model", "unknown")
export type LlamaServerMode = Schema.Schema.Type<typeof LlamaServerMode>
export const LlamaServerOperation = Schema.Literal("health", "models", "props", "apply-template", "events", "load", "unload")
export type LlamaServerOperation = Schema.Schema.Type<typeof LlamaServerOperation>
export const LlamaServerFailureReason = Schema.Literal("transport", "rejected", "invalid-response")
export type LlamaServerFailureReason = Schema.Schema.Type<typeof LlamaServerFailureReason>
export const LlamaServerResponseSchema = Schema.Literal("HealthResponse", "ModelsResponse", "PropsResponse", "ApplyTemplateResponse", "ModelEvent")
export type LlamaServerResponseSchema = Schema.Schema.Type<typeof LlamaServerResponseSchema>
export const LlamaServerMethod = Schema.Literal("GET", "POST")
export type LlamaServerMethod = Schema.Schema.Type<typeof LlamaServerMethod>
export const LlamaRouterControlOperation = Schema.Literal("load", "unload")
export type LlamaRouterControlOperation = Schema.Schema.Type<typeof LlamaRouterControlOperation>

const NonNegative = Schema.Number.pipe(Schema.filter((value) => Number.isFinite(value) && value >= 0))
const Optional = Schema.OptionFromSelf
export const LlamaDiagnosticSchema = Schema.Struct({
  code: Schema.String,
  message: Schema.String,
  modelId: Optional(LlamaServedModelIdSchema),
})
export type LlamaDiagnostic = Schema.Schema.Type<typeof LlamaDiagnosticSchema>
export const LlamaModelLoadProgressSchema = Schema.Struct({
  completed: Optional(Schema.Number),
  total: Optional(Schema.Number),
  fraction: Optional(Schema.Number),
})
export type LlamaModelLoadProgress = Schema.Schema.Type<typeof LlamaModelLoadProgressSchema>
export const LlamaModelFailureSchema = Schema.Struct({
  exitCode: Optional(Schema.Int),
  message: Optional(Schema.String),
})
export type LlamaModelFailure = Schema.Schema.Type<typeof LlamaModelFailureSchema>
export const LlamaServedModelObservationSchema = Schema.Struct({
  id: LlamaServedModelIdSchema,
  status: LlamaServedModelStatus,
  serverDisplayName: Optional(Schema.String),
  reportedModelPath: Optional(NormalizedLlamaModelPath),
  activeContextTokens: Optional(NonNegative),
  architecture: Optional(Schema.String),
  serverFileType: Optional(Schema.String),
  serverReportedSizeBytes: Optional(NonNegative),
  inputModalities: Optional(Schema.Array(Schema.String)),
  outputModalities: Optional(Schema.Array(Schema.String)),
  loadProgress: Optional(LlamaModelLoadProgressSchema),
  failure: Optional(LlamaModelFailureSchema),
})
export type LlamaServedModelObservation = Schema.Schema.Type<typeof LlamaServedModelObservationSchema>
export const LlamaInstanceObservationSchema = Schema.Struct({
  id: LlamaInstanceIdSchema,
  ownership: LlamaInstanceOwnership,
  health: LlamaServerHealth,
  mode: LlamaServerMode,
  build: Optional(Schema.String),
  capabilities: Schema.Struct({
    models: Availability,
    modelEvents: Availability,
    load: Availability,
    unload: Availability,
    sleep: Availability,
  }),
  models: Schema.Array(LlamaServedModelObservationSchema),
  diagnostics: Schema.Array(LlamaDiagnosticSchema),
})
export type LlamaInstanceObservation = Schema.Schema.Type<typeof LlamaInstanceObservationSchema>
export type LlamaServerModelEventTarget = Data.TaggedEnum<{
  AllModels: Record<never, never>
  Model: { readonly id: LlamaServedModelId }
}>
export const LlamaServerModelEventTarget = Data.taggedEnum<LlamaServerModelEventTarget>()
export interface LlamaServerModelEvent {
  readonly target: LlamaServerModelEventTarget
  readonly event: string
  readonly status: Option.Option<LlamaServedModelStatus>
  readonly progress: Option.Option<LlamaModelLoadProgress>
}
export class LlamaServerError extends Data.TaggedError("LlamaServerError")<{
  readonly operation: LlamaServerOperation
  readonly reason: LlamaServerFailureReason
  readonly status: Option.Option<number>
  readonly schema: Option.Option<LlamaServerResponseSchema>
  readonly issues: readonly SchemaIssue[]
}> {}
export interface LlamaServerObserver {
  readonly health: Effect.Effect<LlamaServerHealth, LlamaServerError>
  readonly models: Effect.Effect<readonly LlamaServedModelObservation[], LlamaServerError>
  readonly events: Stream.Stream<LlamaServerModelEvent, LlamaServerError>
  readonly observe: (id: LlamaInstanceId, ownership: LlamaInstanceOwnership) => Effect.Effect<LlamaInstanceObservation, LlamaServerError>
}
export interface LlamaRouterController {
  readonly load: (id: LlamaServedModelId) => Effect.Effect<void, LlamaServerError>
  readonly unload: (id: LlamaServedModelId) => Effect.Effect<void, LlamaServerError>
}
export interface LlamaServerClient {
  readonly observer: LlamaServerObserver
  readonly controller: LlamaRouterController
  readonly props: (model: Option.Option<LlamaServedModelId>) => Effect.Effect<LlamaModelProperties, LlamaServerError>
  readonly applyTemplate: (
    model: LlamaServedModelId,
    request: LlamaApplyTemplateRequest,
  ) => Effect.Effect<LlamaApplyTemplateResponse, LlamaServerError>
}
export interface LlamaServerClientOptions {
  readonly origin: URL
  readonly authorization: Option.Option<Redacted.Redacted<string>>
  readonly timeout: Option.Option<Duration.DurationInput>
}

const StatusValue = LlamaServedModelStatus
const StatusObject = Schema.Struct({
  value: Schema.optional(StatusValue),
  failed: Schema.optional(Schema.Boolean),
  exit_code: Schema.optional(Schema.Int),
  message: Schema.optional(Schema.String),
  args: Schema.optional(Schema.Array(Schema.String)),
})
const Status = Schema.Union(StatusValue, StatusObject)
type Status = Schema.Schema.Type<typeof Status>
const Metadata = Schema.Struct({
  n_ctx: Schema.optional(NonNegative),
  size: Schema.optional(NonNegative),
  ftype: Schema.optional(Schema.String),
  "general.name": Schema.optional(Schema.String),
  general_name: Schema.optional(Schema.String),
  "general.architecture": Schema.optional(Schema.String),
  general_architecture: Schema.optional(Schema.String),
})
const Progress = Schema.Struct({
  completed: Schema.optional(NonNegative),
  total: Schema.optional(NonNegative),
  fraction: Schema.optional(NonNegative),
  value: Schema.optional(NonNegative),
})
type Progress = Schema.Schema.Type<typeof Progress>
const RawArchitecture = Schema.Struct({
  input_modalities: Schema.optional(Schema.Array(Schema.String)),
  output_modalities: Schema.optional(Schema.Array(Schema.String)),
})
type RawArchitecture = Schema.Schema.Type<typeof RawArchitecture>
const RawModel = Schema.Struct({
  id: Schema.optional(LlamaServedModelIdSchema),
  model: Schema.optional(LlamaServedModelIdSchema),
  status: Schema.optional(Status),
  path: Schema.optional(Schema.String),
  model_path: Schema.optional(Schema.String),
  n_ctx: Schema.optional(NonNegative),
  size: Schema.optional(NonNegative),
  ftype: Schema.optional(Schema.String),
  "general.name": Schema.optional(Schema.String),
  general_name: Schema.optional(Schema.String),
  "general.architecture": Schema.optional(Schema.String),
  general_architecture: Schema.optional(Schema.String),
  meta: Schema.optional(Schema.NullOr(Metadata)),
  architecture: Schema.optional(RawArchitecture),
  input_modalities: Schema.optional(Schema.Array(Schema.String)),
  output_modalities: Schema.optional(Schema.Array(Schema.String)),
  progress: Schema.optional(Progress),
  failure: Schema.optional(Schema.Struct({
    exit_code: Schema.optional(Schema.Int),
    message: Schema.optional(Schema.String),
  })),
})
type RawModel = Schema.Schema.Type<typeof RawModel>
const RawModelArray = Schema.Array(RawModel)
const RouterModelsEnvelope = Schema.Struct({ models: RawModelArray })
const DataModelsEnvelope = Schema.Struct({ data: RawModelArray })
const OpenAiModelsResponse = DataModelsEnvelope
const PropsResponse = Schema.Struct({
  build_info: Schema.optional(Schema.String),
  model_path: Schema.optional(Schema.String),
  model_ftype: Schema.optional(Schema.String),
  default_generation_settings: Schema.optional(Schema.Struct({ n_ctx: Schema.optional(NonNegative) })),
  modalities: Schema.optional(Schema.Struct({
    vision: Schema.optional(Schema.Boolean),
    audio: Schema.optional(Schema.Boolean),
    video: Schema.optional(Schema.Boolean),
  })),
  chat_template: Schema.optional(Schema.String),
  chat_template_tool_use: Schema.optional(Schema.String),
})
type PropsResponse = Schema.Schema.Type<typeof PropsResponse>
export const LlamaModelPropertiesSchema = Schema.Struct({
  modelPath: Schema.OptionFromSelf(NormalizedLlamaModelPath),
  contextTokens: Schema.OptionFromSelf(NonNegative),
  modalities: Schema.Struct({
    vision: Schema.Boolean,
    audio: Schema.Boolean,
    video: Schema.Boolean,
  }),
  build: Schema.OptionFromSelf(Schema.String),
  chatTemplate: Schema.OptionFromSelf(Schema.String),
  chatTemplateToolUse: Schema.OptionFromSelf(Schema.String),
})
export type LlamaModelProperties = typeof LlamaModelPropertiesSchema.Type

export interface LlamaApplyTemplateRequest {
  readonly messages: readonly Record<string, unknown>[]
  readonly tools?: readonly Record<string, unknown>[]
  readonly toolChoice?: unknown
  readonly chatTemplateKwargs?: Readonly<Record<string, unknown>>
}
const ApplyTemplateResponse = Schema.Struct({ prompt: Schema.String })
export type LlamaApplyTemplateResponse = typeof ApplyTemplateResponse.Type
const HealthResponse = Schema.Struct({ status: Schema.Literal("ok") })
const ModelEvent = Schema.parseJson(Schema.Struct({
  model: Schema.Union(Schema.Literal("*"), LlamaServedModelIdSchema),
  event: Schema.String,
  data: Schema.optional(Schema.Struct({
    status: Schema.optional(Status),
    progress: Schema.optional(Progress),
  })),
}))

type ModelListing = Data.TaggedEnum<{
  Router: { readonly models: readonly RawModel[] }
  Single: { readonly models: readonly RawModel[] }
}>
const ModelListing = Data.taggedEnum<ModelListing>()
const noIssues: readonly SchemaIssue[] = []

const firstSome = <A>(values: readonly Option.Option<A>[]): Option.Option<A> => {
  for (const value of values) if (Option.isSome(value)) return value
  return Option.none()
}
const optionalText = (value: string | undefined): Option.Option<string> => pipe(
  Option.fromNullable(value),
  Option.map((text) => text.trim()),
  Option.filter((text) => text.length > 0 && text !== "none"),
)
const statusFrom = (value: Option.Option<Status>, absent: LlamaServedModelStatus): LlamaServedModelStatus => Option.match(value, {
  onNone: () => absent,
  onSome: (status) => Schema.is(StatusValue)(status)
    ? status
    : Option.getOrElse(Option.fromNullable(status.value), () => "unknown"),
})
const failureFromStatus = (status: Option.Option<Status>): Option.Option<LlamaModelFailure> => pipe(
  status,
  Option.filter((value) => !Schema.is(StatusValue)(value) && value.failed === true),
  Option.map((value) => {
    if (Schema.is(StatusValue)(value)) return { exitCode: Option.none(), message: Option.none() }
    return { exitCode: Option.fromNullable(value.exit_code), message: optionalText(value.message) }
  }),
)
const progressFrom = (progress: Option.Option<Progress>): Option.Option<LlamaModelLoadProgress> => pipe(
  progress,
  Option.map((value) => ({
    completed: Option.fromNullable(value.completed),
    total: Option.fromNullable(value.total),
    fraction: firstSome([Option.fromNullable(value.fraction), Option.fromNullable(value.value)]),
  })),
)

const metadataName = (metadata: Option.Option<Schema.Schema.Type<typeof Metadata>>): Option.Option<string> => pipe(
  metadata,
  Option.flatMap((value) => firstSome([optionalText(value["general.name"]), optionalText(value.general_name)])),
)
const metadataArchitecture = (metadata: Option.Option<Schema.Schema.Type<typeof Metadata>>): Option.Option<string> => pipe(
  metadata,
  Option.flatMap((value) => firstSome([optionalText(value["general.architecture"]), optionalText(value.general_architecture)])),
)
const metadataFileType = (metadata: Option.Option<Schema.Schema.Type<typeof Metadata>>): Option.Option<string> => pipe(
  metadata,
  Option.flatMap((value) => optionalText(value.ftype)),
)
const propsContextSize = (props: Option.Option<PropsResponse>): Option.Option<number> => pipe(
  props,
  Option.flatMap((value) => Option.fromNullable(value.default_generation_settings)),
  Option.flatMap((settings) => Option.fromNullable(settings.n_ctx)),
)
const architectureInputModalities = (architecture: Option.Option<RawArchitecture>): Option.Option<readonly string[]> => pipe(
  architecture,
  Option.flatMap((value) => Option.fromNullable(value.input_modalities)),
)
const architectureOutputModalities = (architecture: Option.Option<RawArchitecture>): Option.Option<readonly string[]> => pipe(
  architecture,
  Option.flatMap((value) => Option.fromNullable(value.output_modalities)),
)
const eventTarget = (model: Schema.Schema.Type<typeof ModelEvent>["model"]): LlamaServerModelEventTarget => model === "*"
  ? LlamaServerModelEventTarget.AllModels()
  : LlamaServerModelEventTarget.Model({ id: model })

export const makeLlamaServerClient = (
  options: LlamaServerClientOptions,
): Effect.Effect<LlamaServerClient, never, HttpClient.HttpClient> => Effect.gen(function* () {
  const client = yield* HttpClient.HttpClient
  const authorization = options.authorization
  const timeout: Duration.DurationInput = Option.getOrElse(options.timeout, () => "10 seconds" as const)

  const request = (
    operation: LlamaServerOperation,
    route: string,
    method: LlamaServerMethod = "GET",
    body: Option.Option<object> = Option.none(),
  ) => Effect.gen(function* () {
    const url = new URL(route, options.origin).toString()
    const base = method === "GET" ? HttpClientRequest.get(url) : HttpClientRequest.post(url)
    const withBody = Option.match(body, {
      onNone: () => base,
      onSome: (value) => HttpClientRequest.bodyUnsafeJson(base, value),
    })
    const accepted = HttpClientRequest.acceptJson(withBody)
    const authenticated = Option.match(authorization, {
      onNone: () => accepted,
      onSome: (secret) => HttpClientRequest.bearerToken(accepted, Redacted.value(secret)),
    })
    const response = yield* client.execute(authenticated).pipe(
      Effect.timeout(timeout),
      Effect.mapError(() => new LlamaServerError({ operation, reason: "transport", status: Option.none(), schema: Option.none(), issues: noIssues })),
    )
    if (response.status < 200 || response.status >= 300) {
      return yield* new LlamaServerError({ operation, reason: "rejected", status: Option.some(response.status), schema: Option.none(), issues: noIssues })
    }
    return response
  })

  const json = <A, I>(
    operation: LlamaServerOperation,
    schemaName: LlamaServerResponseSchema,
    schema: Schema.Schema<A, I>,
    route: string,
    method: LlamaServerMethod = "GET",
    body: Option.Option<object> = Option.none(),
  ) => request(operation, route, method, body).pipe(
    Effect.flatMap((response) => response.json.pipe(
      Effect.mapError(() => new LlamaServerError({ operation, reason: "invalid-response", status: Option.none(), schema: Option.some(schemaName), issues: noIssues })),
    )),
    Effect.flatMap((value) => Schema.decodeUnknown(schema)(value).pipe(
      Effect.mapError((error) => new LlamaServerError({ operation, reason: "invalid-response", status: Option.none(), schema: Option.some(schemaName), issues: formatSchemaIssues(error) })),
    )),
  )

  const props = (model: Option.Option<LlamaServedModelId>) => {
    const suffix = Option.match(model, {
      onNone: () => "?autoload=false",
      onSome: (id) => `?model=${encodeURIComponent(id)}&autoload=false`,
    })
    return json("props", "PropsResponse", PropsResponse, `/props${suffix}`).pipe(
      Effect.catchTag("LlamaServerError", (error) => Option.contains(error.status, 404)
        ? json("props", "PropsResponse", PropsResponse, `/v1/props${suffix}`)
        : Effect.fail(error)),
    )
  }

  const modelProperties = (model: Option.Option<LlamaServedModelId>) => props(model).pipe(
    Effect.map((value): LlamaModelProperties => ({
      modelPath: pipe(optionalText(value.model_path), Option.flatMap((path) => Option.fromNullable(normalizeLlamaModelPath(path)))),
      contextTokens: propsContextSize(Option.some(value)),
      modalities: {
        vision: value.modalities?.vision === true,
        audio: value.modalities?.audio === true,
        video: value.modalities?.video === true,
      },
      build: optionalText(value.build_info),
      chatTemplate: Option.fromNullable(value.chat_template),
      chatTemplateToolUse: Option.fromNullable(value.chat_template_tool_use),
    })),
  )

  const applyTemplate = (model: LlamaServedModelId, templateRequest: LlamaApplyTemplateRequest) => {
    const query = `?model=${encodeURIComponent(model)}&autoload=false`
    const body = {
      model,
      messages: [...templateRequest.messages],
      ...(templateRequest.tools === undefined ? {} : { tools: [...templateRequest.tools] }),
      ...(templateRequest.toolChoice === undefined ? {} : { tool_choice: templateRequest.toolChoice }),
      ...(templateRequest.chatTemplateKwargs === undefined ? {} : { chat_template_kwargs: templateRequest.chatTemplateKwargs }),
    }
    return json("apply-template", "ApplyTemplateResponse", ApplyTemplateResponse, `/apply-template${query}`, "POST", Option.some(body)).pipe(
      Effect.catchTag("LlamaServerError", (error) => Option.contains(error.status, 404)
        ? json("apply-template", "ApplyTemplateResponse", ApplyTemplateResponse, `/v1/apply-template${query}`, "POST", Option.some(body))
        : Effect.fail(error)),
    )
  }

  const health = request("health", "/health").pipe(
    Effect.flatMap((response) => response.json.pipe(
      Effect.mapError(() => new LlamaServerError({ operation: "health", reason: "invalid-response", status: Option.none(), schema: Option.some("HealthResponse"), issues: noIssues })),
      Effect.flatMap((body) => Schema.decodeUnknown(HealthResponse)(body).pipe(
        Effect.mapError((error) => new LlamaServerError({ operation: "health", reason: "invalid-response", status: Option.none(), schema: Option.some("HealthResponse"), issues: formatSchemaIssues(error) })),
      )),
      Effect.as("ready" as const),
    )),
    Effect.catchTag("LlamaServerError", (error) => Option.contains(error.status, 503)
      ? Effect.succeed("loading" as const)
      : Effect.fail(error)),
  )

  const decodeRouterListing = (value: unknown): Effect.Effect<ModelListing, LlamaServerError> => {
    const direct = Schema.decodeUnknownOption(RawModelArray)(value)
    if (Option.isSome(direct)) return Effect.succeed(ModelListing.Router({ models: direct.value }))
    const modelsEnvelope = Schema.decodeUnknownOption(RouterModelsEnvelope)(value)
    if (Option.isSome(modelsEnvelope)) return Effect.succeed(ModelListing.Router({ models: modelsEnvelope.value.models }))
    const dataEnvelope = Schema.decodeUnknownOption(DataModelsEnvelope)(value)
    if (Option.isSome(dataEnvelope)) return Effect.succeed(ModelListing.Router({ models: dataEnvelope.value.data }))
    return Effect.fail(new LlamaServerError({ operation: "models", reason: "invalid-response", status: Option.none(), schema: Option.some("ModelsResponse"), issues: noIssues }))
  }

  const routerModels = request("models", "/models").pipe(
    Effect.flatMap((response) => response.json.pipe(
      Effect.mapError(() => new LlamaServerError({ operation: "models", reason: "invalid-response", status: Option.none(), schema: Option.some("ModelsResponse"), issues: noIssues })),
    )),
    Effect.flatMap(decodeRouterListing),
  )
  const rawModels = routerModels.pipe(
    Effect.catchTag("LlamaServerError", (error) => Option.contains(error.status, 404)
      ? json("models", "ModelsResponse", OpenAiModelsResponse, "/v1/models").pipe(
        Effect.map(({ data }) => ModelListing.Single({ models: data })),
      )
      : Effect.fail(error)),
  )

  const normalizeModel = (listing: ModelListing, raw: RawModel): Effect.Effect<LlamaServedModelObservation, LlamaServerError> => Effect.gen(function* () {
    const identifier = firstSome([Option.fromNullable(raw.id), Option.fromNullable(raw.model)])
    if (Option.isNone(identifier)) {
      return yield* new LlamaServerError({ operation: "models", reason: "invalid-response", status: Option.none(), schema: Option.some("ModelsResponse"), issues: noIssues })
    }
    const status = statusFrom(Option.fromNullable(raw.status), listing._tag === "Router" ? "unknown" : "loaded")
    const modelProps = status === "loaded" || status === "sleeping"
      ? yield* props(Option.some(identifier.value)).pipe(Effect.option)
      : Option.none<PropsResponse>()
    const meta = Option.fromNullable(raw.meta)
    const serverFailure = pipe(
      Option.fromNullable(raw.failure),
      Option.map((failure): LlamaModelFailure => ({
        exitCode: Option.fromNullable(failure.exit_code),
        message: optionalText(failure.message),
      })),
    )
    const failure = Option.orElse(serverFailure, () => failureFromStatus(Option.fromNullable(raw.status)))
    const propsFileType = pipe(modelProps, Option.flatMap((value) => optionalText(value.model_ftype)))
    const reportedModelPath = firstSome([
      pipe(modelProps, Option.flatMap((value) => optionalText(value.model_path))),
    ]).pipe(Option.flatMap((value) => Option.fromNullable(normalizeLlamaModelPath(value))))
    return {
      id: identifier.value,
      status: Option.isSome(failure) ? "failed" : status,
      serverDisplayName: firstSome([optionalText(raw["general.name"]), optionalText(raw.general_name), metadataName(meta)]),
      reportedModelPath,
      activeContextTokens: firstSome([
        Option.fromNullable(raw.n_ctx),
        pipe(meta, Option.flatMap((value) => Option.fromNullable(value.n_ctx))),
        propsContextSize(modelProps),
      ]),
      architecture: firstSome([optionalText(raw["general.architecture"]), optionalText(raw.general_architecture), metadataArchitecture(meta)]),
      serverFileType: firstSome([optionalText(raw.ftype), metadataFileType(meta), propsFileType]),
      serverReportedSizeBytes: firstSome([Option.fromNullable(raw.size), pipe(meta, Option.flatMap((value) => Option.fromNullable(value.size)))]),
      inputModalities: firstSome([Option.fromNullable(raw.input_modalities), architectureInputModalities(Option.fromNullable(raw.architecture))]),
      outputModalities: firstSome([Option.fromNullable(raw.output_modalities), architectureOutputModalities(Option.fromNullable(raw.architecture))]),
      loadProgress: progressFrom(Option.fromNullable(raw.progress)),
      failure,
    }
  })

  const modelsWithMode = rawModels.pipe(
    Effect.flatMap((listing) => Effect.forEach(
      listing.models,
      (raw) => normalizeModel(listing, raw),
      { concurrency: 4 },
    ).pipe(Effect.map((models) => ({ listing, models })))),
  )
  const models = modelsWithMode.pipe(Effect.map(({ models }) => models))
  const events = request("events", "/models/sse?autoload=false").pipe(
    Stream.fromEffect,
    Stream.flatMap((response) => response.stream.pipe(
      Stream.mapError(() => new LlamaServerError({ operation: "events", reason: "transport", status: Option.none(), schema: Option.none(), issues: noIssues })),
    )),
    Stream.decodeText(),
    Stream.splitLines,
    Stream.filter((line) => line.startsWith("data:")),
    Stream.map((line) => line.slice(5).trim()),
    Stream.filter((line) => line.length > 0),
    Stream.mapEffect((line) => Schema.decode(ModelEvent)(line).pipe(
      Effect.mapError((error) => new LlamaServerError({ operation: "events", reason: "invalid-response", status: Option.none(), schema: Option.some("ModelEvent"), issues: formatSchemaIssues(error) })),
      Effect.map((event): LlamaServerModelEvent => {
        const data = Option.fromNullable(event.data)
        return {
          target: eventTarget(event.model),
          event: event.event,
          status: pipe(data, Option.flatMap((value) => Option.fromNullable(value.status)), Option.map((value) => statusFrom(Option.some(value), "unknown"))),
          progress: pipe(data, Option.flatMap((value) => progressFrom(Option.fromNullable(value.progress)))),
        }
      }),
    )),
  )
  const control = (operation: LlamaRouterControlOperation, id: LlamaServedModelId) => request(
    operation,
    `/models/${operation}`,
    "POST",
    Option.some({ model: id }),
  ).pipe(Effect.asVoid)

  return {
    props: modelProperties,
    applyTemplate,
    observer: {
      health,
      models,
      events,
      observe: (id, ownership) => Effect.gen(function* () {
        const observedHealth = yield* health
        const modelResult = yield* Effect.either(modelsWithMode)
        const listed: readonly LlamaServedModelObservation[] = modelResult._tag === "Right"
          ? modelResult.right.models
          : []
        const router = modelResult._tag === "Right"
          && modelResult.right.listing._tag === "Router"

        let build = Option.none<string>()
        if (modelResult._tag === "Right" && listed.length === 1) {
          build = yield* props(Option.none()).pipe(
            Effect.map((value) => optionalText(value.build_info)),
            Effect.orElseSucceed(() => Option.none()),
          )
        }

        let modelAvailability: Availability = "supported"
        const diagnostics: LlamaDiagnostic[] = []

        if (modelResult._tag === "Left") {
          modelAvailability = Option.contains(modelResult.left.status, 404)
            ? "unsupported"
            : "unknown"
          const statusSuffix = Option.match(modelResult.left.status, {
            onNone: () => "",
            onSome: (status) => ` (${status})`,
          })
          diagnostics.push({
            code: "models_unavailable",
            message: `${modelResult.left.reason}${statusSuffix}`,
            modelId: Option.none(),
          })
        }
        for (const model of listed) {
          if ((model.status === "loaded" || model.status === "sleeping") && Option.isNone(model.reportedModelPath)) {
            diagnostics.push({
              code: "model_path_unavailable",
              message: "Loaded model did not provide a valid model-scoped model_path.",
              modelId: Option.some(model.id),
            })
          }
        }

        let mode: LlamaInstanceObservation["mode"] = "unknown"
        if (modelResult._tag === "Right") {
          mode = router ? "router" : "single-model"
        }

        return {
          id,
          ownership,
          health: observedHealth,
          mode,
          build,
          capabilities: {
            models: modelAvailability,
            modelEvents: router ? "supported" : "unknown",
            load: router ? "supported" : "unknown",
            unload: router ? "supported" : "unknown",
            sleep: "unknown",
          },
          models: listed,
          diagnostics,
        }
      }),
    },
    controller: {
      load: (id) => control("load", id),
      unload: (id) => control("unload", id),
    },
  }
})
