import { Array as Arr, Cause, Context, Deferred, Effect, Layer, Option, Ref, Scope, Stream } from "effect"
import { IcnInventory, IcnRecipes, type Generated } from "@magnitudedev/icn"
import {
  LocalInferenceError,
} from "@magnitudedev/protocol"
import {
  LocalModelConfiguration,
  type ModelSlotsConfiguration,
} from "../model-configuration"
import { reconcileSelectedServingConfiguration } from "./serving-configuration"
import { AcnActivityTracker } from "../activity-tracker"

const PROFILE_PARALLEL_SEQUENCES = 1
const MAX_ADMISSION_HISTORY = 256
const EMPTY_MODEL_SLOTS: ModelSlotsConfiguration = {}
type CommandAdmission = { readonly operationId: string }

export interface IcnCommandsService {
  readonly downloadModel: (
    configurationId: string,
    requestId: string,
  ) => Effect.Effect<{ readonly operationId: string }, LocalInferenceError>
  readonly activateModel: (
    selectionId: string,
    requestId: string,
  ) => Effect.Effect<{ readonly operationId: string }, LocalInferenceError>
  readonly deleteModel: (selectionId: string) => Effect.Effect<void, LocalInferenceError>
  readonly restart: (requestId: string) => Effect.Effect<{ readonly operationId: string }, LocalInferenceError>
  readonly disable: Effect.Effect<void, LocalInferenceError>
}

export class IcnCommands extends Context.Tag("Acn/IcnCommands")<
  IcnCommands,
  IcnCommandsService
>() {}

const commandError = (
  code: LocalInferenceError["code"],
  operation: string,
  cause: unknown,
  retryable = true,
) => new LocalInferenceError({
  code,
  operation,
  message: cause instanceof Error ? cause.message : String(cause),
  retryable,
})

export const makeIcnCommands = (): Layer.Layer<
  IcnCommands,
  never,
  IcnInventory | IcnRecipes | LocalModelConfiguration | AcnActivityTracker
> => Layer.scoped(
  IcnCommands,
  Effect.gen(function* () {
    const inventory = yield* IcnInventory
    const recipes = yield* IcnRecipes
    const configuration = yield* LocalModelConfiguration
    const activity = yield* AcnActivityTracker
    const serviceScope = yield* Scope.Scope
    const admissionLock = yield* Effect.makeSemaphore(1)
    const admissions = yield* Ref.make<ReadonlyMap<string, Deferred.Deferred<CommandAdmission, LocalInferenceError>>>(
      new Map(),
    )

    yield* reconcileSelectedServingConfiguration(inventory, configuration).pipe(
      Effect.catchAll((cause) => Effect.logWarning("Unable to reapply local model serving configuration").pipe(
        Effect.annotateLogs({ cause: String(cause) }),
      )),
    )

    const admit = (
      requestId: string,
      start: Effect.Effect<CommandAdmission, LocalInferenceError>,
    ): Effect.Effect<CommandAdmission, LocalInferenceError> => Effect.gen(function* () {
      const admission = yield* admissionLock.withPermits(1)(Effect.gen(function* () {
        const current = yield* Ref.get(admissions)
        const existing = Option.fromNullable(current.get(requestId))
        if (Option.isSome(existing)) return { deferred: existing.value, created: false }
        const deferred = yield* Deferred.make<CommandAdmission, LocalInferenceError>()
        const next = new Map(current).set(requestId, deferred)
        while (next.size > MAX_ADMISSION_HISTORY) {
          const oldest = next.keys().next()
          if (oldest.done) break
          next.delete(oldest.value)
        }
        yield* Ref.set(admissions, next)
        return { deferred, created: true }
      }))
      if (admission.created) {
        yield* Effect.forkIn(
          Deferred.complete(admission.deferred, start).pipe(Effect.asVoid),
          serviceScope,
        )
      }
      return yield* Deferred.await(admission.deferred)
    })

    const modelForSelection = (selectionId: string) => Effect.gen(function* () {
      const models = (yield* inventory.get).state.data
      const direct = Arr.findFirst(models, (model) => model.id === selectionId)
      if (Option.isSome(direct)) return direct
      const configuredId = Option.flatMap(
        Option.fromNullable((yield* configuration.get).selectedProfile),
        (selected) => selected.configurationId === selectionId
          ? Option.fromNullable(selected.providerModelId)
          : Option.none(),
      )
      return Option.flatMap(
        configuredId,
        (providerModelId) => Arr.findFirst(models, (model) => model.id === providerModelId),
      )
    })

    const admittedOperation = <Event extends { readonly operation_id: string }>(
      activityName: string,
      events: Stream.Stream<Event, unknown>,
      onEvent: (event: Event) => Effect.Effect<void>,
      failure: (cause: unknown) => LocalInferenceError,
    ) => Effect.gen(function* () {
      const admitted = yield* Deferred.make<string, LocalInferenceError>()
      const consume = events.pipe(
        Stream.runForEach((event) => Deferred.succeed(admitted, event.operation_id).pipe(
          Effect.zipRight(onEvent(event)),
        )),
        Effect.catchAllCause((cause) => Deferred.fail(
          admitted,
          failure(Option.getOrElse(Cause.failureOption(cause), () => Cause.pretty(cause))),
        ).pipe(Effect.zipRight(Effect.logError(`ICN ${activityName} stream failed`).pipe(
          Effect.annotateLogs({ cause: Cause.pretty(cause) }),
        )))),
        Effect.ensuring(Deferred.fail(
          admitted,
          failure("ICN ended the operation stream before admitting an operation."),
        ).pipe(Effect.ignore)),
      )
      yield* Effect.forkIn(activity.withActiveWork(activityName, consume), serviceScope)
      return { operationId: yield* Deferred.await(admitted) }
    })

    const downloadModel: IcnCommandsService["downloadModel"] = (configurationId, requestId) =>
      admit(requestId, Effect.gen(function* () {
        const recommendation = yield* recipes.resolve(configurationId)
        if (Option.isNone(recommendation)) {
          return yield* commandError(
            "invalid_selection",
            "download local model",
            "The selected recommendation is no longer available.",
            false,
          )
        }
        yield* configuration.selectProfile({
          configurationId,
          catalogModelId: recommendation.value.catalogModelId,
          contextTokens: recommendation.value.contextTokens,
        }).pipe(Effect.mapError((cause) => commandError(
          "configuration_failed",
          "save local model selection",
          cause,
        )))
        const request: Generated.DownloadModelRequestSchema = {
          source: {
            type: "hugging_face",
            repository: recommendation.value.repo,
            revision: recommendation.value.revision,
          },
          components: recommendation.value.files.map((file, index) => ({
            path: file.path,
            role: file.role,
            expected_sha256: file.sha256 ? Option.some(file.sha256) : Option.none(),
            shard_index: index === 0 ? Option.none() : Option.some(index),
          })),
          relationships: [],
          serving_profile: {
            context_length: recommendation.value.contextTokens,
            parallel_sequences: PROFILE_PARALLEL_SEQUENCES,
          },
        }
        const response = yield* inventory.downloadModel({ payload: request }).pipe(
          Effect.mapError((cause) => commandError("configuration_failed", "download local model", cause)),
        )
        return yield* admittedOperation(
          "local-inference:download",
          response.events,
          (event) => event.type === "ready"
            ? configuration.selectProfile({
                configurationId,
                catalogModelId: recommendation.value.catalogModelId,
                contextTokens: recommendation.value.contextTokens,
                providerModelId: event.model.id,
              }).pipe(Effect.ignore)
            : Effect.void,
          (cause) => commandError("configuration_failed", "download local model", cause),
        )
      }))

    const activateModel: IcnCommandsService["activateModel"] = (selectionId, requestId) =>
      admit(requestId, Effect.gen(function* () {
        const model = yield* modelForSelection(selectionId).pipe(
          Effect.mapError((cause) => commandError("artifact_unavailable", "activate local model", cause)),
        )
        if (Option.isNone(model)) {
          return yield* commandError(
            "artifact_unavailable",
            "activate local model",
            "The selected model is not available in ICN.",
            false,
          )
        }
        yield* reconcileSelectedServingConfiguration(inventory, configuration, Option.some([model.value])).pipe(
          Effect.mapError((cause) => commandError("configuration_failed", "configure local model serving", cause)),
        )
        const response = yield* inventory.loadModel({ path: { model_id: model.value.id } }).pipe(
          Effect.mapError((cause) => commandError("runtime_start_failed", "activate local model", cause)),
        )
        return yield* admittedOperation(
          "local-inference:activate",
          response.events,
          () => Effect.void,
          (cause) => commandError("runtime_start_failed", "activate local model", cause),
        )
      }))

    const deleteModel: IcnCommandsService["deleteModel"] = (selectionId) => Effect.gen(function* () {
      const model = yield* modelForSelection(selectionId).pipe(
        Effect.mapError((cause) => commandError("artifact_unavailable", "delete local model", cause)),
      )
      if (Option.isNone(model)) {
        return yield* commandError(
          "artifact_unavailable",
          "delete local model",
          "The selected model is not available in ICN.",
          false,
        )
      }
      yield* inventory.deleteModel({
        path: { model_id: model.value.id },
        urlParams: { dry_run: Option.none() },
      }).pipe(Effect.mapError((cause) => commandError("artifact_active", "delete local model", cause)))
    })

    const disable = Effect.gen(function* () {
      const models = yield* configuration.getModels.pipe(
        Effect.mapError((cause) => commandError("configuration_failed", "read model slots", cause)),
      )
      const slots = Option.getOrElse(
        Option.fromNullable(models.slots),
        () => EMPTY_MODEL_SLOTS,
      )
      const updates = Object.fromEntries(
        (["primary", "secondary"] as const)
          .filter((slotId) => Option.exists(
            Option.fromNullable(slots[slotId]),
            (slot) => slot.providerId === "local",
          ))
          .map((slotId) => [slotId, {}]),
      )
      if (Object.keys(updates).length > 0) {
        yield* configuration.updateSlots(updates).pipe(
          Effect.mapError((cause) => commandError("configuration_failed", "clear local model slots", cause)),
        )
      }
      const modelsToUnload = (yield* inventory.get).state.data
        .filter((model) => model.residency.type === "loaded")
      yield* Effect.forEach(
        modelsToUnload,
        (model) => inventory.unloadModel({ path: { model_id: model.id } }),
        { discard: true },
      ).pipe(Effect.mapError((cause) => commandError(
        "configuration_failed",
        "disable local inference",
        cause,
      )))
    })

    const restart: IcnCommandsService["restart"] = (requestId) => admit(requestId, Effect.gen(function* () {
      const current = Arr.findFirst(
        (yield* inventory.get).state.data,
        (model) => model.residency.type === "loaded",
      )
      if (Option.isNone(current)) return { operationId: requestId }
      yield* inventory.unloadModel({ path: { model_id: current.value.id } }).pipe(
        Effect.mapError((cause) => commandError(
          "runtime_start_failed",
          "unload local inference for restart",
          cause,
        )),
      )
      const response = yield* inventory.loadModel({ path: { model_id: current.value.id } }).pipe(
        Effect.mapError((cause) => commandError("runtime_start_failed", "restart local inference", cause)),
      )
      return yield* admittedOperation(
        "local-inference:restart",
        response.events,
        () => Effect.void,
        (cause) => commandError("runtime_start_failed", "restart local inference", cause),
      )
    }))

    return IcnCommands.of({ downloadModel, activateModel, deleteModel, restart, disable })
  }),
)
