import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option, Schema } from "effect"
import {
  LocalInferenceHardwareMirror,
  LocalModelsMirror,
  ModelSlotsMirror,
  OnboardingMirror,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogMirror,
  type ModelOfferingTargetId,
  type ProviderModelId,
  type ReasoningEffort,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredStateAtom } from "./use-mirrored-state"
import { useModelSlotActions } from "./use-local-inference-state"

interface OnboardingModelChoice {
  readonly providerModelId: ProviderModelId
  readonly displayName: string
  readonly reasoningEffort: ReasoningEffort
}

export type OnboardingLoadModelChoice = OnboardingModelChoice

export interface OnboardingDownloadModelChoice extends OnboardingModelChoice {
  readonly targetId: ModelOfferingTargetId
}

export class OnboardingModelCommandFailed extends Schema.TaggedError<
  OnboardingModelCommandFailed
>()("OnboardingModelCommandFailed", {
  command: Schema.Literal("download", "assign", "load", "complete", "cancel", "clear"),
  message: Schema.String,
}) {}

type ActiveOnboardingModelIntent =
  | { readonly _tag: "Loading"; readonly choice: OnboardingLoadModelChoice }
  | { readonly _tag: "Downloading"; readonly choice: OnboardingDownloadModelChoice }

export type OnboardingModelWorkflowIntent =
  | { readonly _tag: "Idle" }
  | ActiveOnboardingModelIntent
  | { readonly _tag: "Cancelling"; readonly operation: ActiveOnboardingModelIntent }

type OnboardingModelWorkflowCommand =
  | { readonly _tag: "Load"; readonly choice: OnboardingLoadModelChoice }
  | { readonly _tag: "DownloadThenLoad"; readonly choice: OnboardingDownloadModelChoice }
  | { readonly _tag: "Cancel" }

export const useOnboardingModelSetup = () => {
  const client = useAgentClient()
  const hardwareAtom = useMirroredStateAtom(LocalInferenceHardwareMirror)
  const modelsAtom = useMirroredStateAtom(LocalModelsMirror)
  const catalogAtom = useMirroredStateAtom(ProviderModelCatalogMirror)
  const slotsAtom = useMirroredStateAtom(ModelSlotsMirror)
  const slotActions = useModelSlotActions()
  const mutations = useMemo(() => ({
    download: client.mutation("DownloadModel"),
    assign: client.mutation("AssignSlot"),
    load: client.mutation("LoadModel"),
    complete: client.mutation("UpdateOnboardingState"),
    cancel: client.mutation("CancelModelDownload"),
    clear: client.mutation("ClearSlot"),
  }), [client])
  const intentAtom = useMemo(
    () => Atom.make<OnboardingModelWorkflowIntent>({ _tag: "Idle" }),
    [],
  )

  const workflowAtom = useMemo(
    () => Atom.fn<OnboardingModelWorkflowCommand>()((command, get) => {
      if (command._tag === "Cancel") {
        const intent = get(intentAtom)
        if (intent._tag === "Idle") return Effect.void
        const operation = intent._tag === "Cancelling" ? intent.operation : intent
        const choice = operation.choice
        get.set(intentAtom, { _tag: "Cancelling", operation })
        const cancelTargetId = operation._tag === "Downloading"
          && Option.exists(
            Result.value(get(modelsAtom)),
            ({ state }) => state.models.some(({ targetId, download }) =>
              targetId === operation.choice.targetId && download._tag === "Downloading"),
          )
          ? operation.choice.targetId
          : null
        const shouldClearSlot = Option.exists(
          Result.value(get(slotsAtom)),
          ({ state }) => {
            const primary = state.slots.primary
            return primary._tag === "ConfiguredLocal"
              && primary.selection.providerId === "local"
              && primary.selection.providerModelId === choice.providerModelId
          },
        )
        return Effect.gen(function* () {
          if (cancelTargetId !== null) {
            yield* get.setResult(mutations.cancel, {
              payload: { targetId: cancelTargetId },
              reactivityKeys: [LocalModelsMirror.id],
            }).pipe(
              Effect.mapError((error) => new OnboardingModelCommandFailed({
                command: "cancel",
                message: error.message,
              })),
            )
          }
          if (shouldClearSlot) {
            yield* get.setResult(mutations.clear, {
              payload: { slotId: PRIMARY_SLOT_ID },
              reactivityKeys: [ModelSlotsMirror.id],
            }).pipe(
              Effect.mapError((error) => new OnboardingModelCommandFailed({
                command: "clear",
                message: error.message,
              })),
            )
          }
        }).pipe(
          Effect.ensuring(Effect.sync(() => get.set(intentAtom, { _tag: "Idle" }))),
        )
      }

      const choice = command.choice
      const activate = Effect.gen(function* () {
        yield* get.setResult(mutations.assign, {
          payload: {
            slotId: PRIMARY_SLOT_ID,
            selection: {
              providerId: ProviderIdSchema.make("local"),
              providerModelId: choice.providerModelId,
              reasoningEffort: choice.reasoningEffort,
            },
          },
          reactivityKeys: [ModelSlotsMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "assign",
            message: error.message,
          })),
        )
        const load = yield* get.setResult(mutations.load, {
          payload: { slotId: PRIMARY_SLOT_ID },
          reactivityKeys: [ModelSlotsMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "load",
            message: error.message,
          })),
        )
        if (load._tag === "Cancelled") return load
        yield* get.setResult(mutations.complete, {
          payload: { completed: true },
          reactivityKeys: [OnboardingMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "complete",
            message: error.message,
          })),
        )
        return load
      })

      if (command._tag === "Load") {
        get.set(intentAtom, { _tag: "Loading", choice })
        return activate
      }

      const downloadChoice = command.choice
      get.set(intentAtom, { _tag: "Downloading", choice: downloadChoice })
      return Effect.gen(function* () {
        yield* get.setResult(mutations.download, {
          payload: { targetId: downloadChoice.targetId },
          reactivityKeys: [LocalModelsMirror.id, ProviderModelCatalogMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "download",
            message: error.message,
          })),
        )
        get.set(intentAtom, { _tag: "Loading", choice })
        return yield* activate
      })
    }),
    [intentAtom, modelsAtom, mutations, slotsAtom],
  )
  const runWorkflow = useAtomSet(workflowAtom)
  const setupAtom = useMemo(
    () => Atom.make((get) => ({
      hardware: Result.map(get(hardwareAtom), ({ state }) => state),
      models: Result.map(get(modelsAtom), ({ state }) => state),
      catalog: Result.map(get(catalogAtom), ({ state }) => state),
      slots: Result.map(get(slotsAtom), ({ state }) => state),
      workflowResult: get(workflowAtom),
      intent: get(intentAtom),
      cancelling: get(intentAtom)._tag === "Cancelling",
    })),
    [catalogAtom, hardwareAtom, intentAtom, modelsAtom, slotsAtom, workflowAtom],
  )
  const setup = useAtomValue(setupAtom)

  const load = useCallback((choice: OnboardingLoadModelChoice) => {
    if (Result.isWaiting(setup.workflowResult)) return
    runWorkflow({ _tag: "Load", choice })
  }, [runWorkflow, setup.workflowResult])

  const downloadThenLoad = useCallback((choice: OnboardingDownloadModelChoice) => {
    if (Result.isWaiting(setup.workflowResult)) return
    runWorkflow({ _tag: "DownloadThenLoad", choice })
  }, [runWorkflow, setup.workflowResult])

  const cancel = useCallback(() => {
    runWorkflow({ _tag: "Cancel" })
  }, [runWorkflow])

  return {
    ...setup,
    slotActions,
    load,
    downloadThenLoad,
    cancel,
  }
}
