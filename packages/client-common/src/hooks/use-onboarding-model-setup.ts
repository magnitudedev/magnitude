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

export interface OnboardingModelChoice {
  readonly targetId: ModelOfferingTargetId
  readonly providerModelId: ProviderModelId
  readonly reasoningEffort: ReasoningEffort
}

export class OnboardingModelCommandFailed extends Schema.TaggedError<
  OnboardingModelCommandFailed
>()("OnboardingModelCommandFailed", {
  command: Schema.Literal("download", "assign", "load", "complete", "cancel", "clear"),
  message: Schema.String,
}) {}

type OnboardingModelWorkflowCommand =
  | { readonly _tag: "Select"; readonly choice: OnboardingModelChoice }
  | { readonly _tag: "Cancel" }

type OnboardingModelWorkflowIntent =
  | { readonly _tag: "Idle" }
  | { readonly _tag: "Selecting"; readonly choice: OnboardingModelChoice }
  | { readonly _tag: "Cancelling"; readonly choice: OnboardingModelChoice }

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
        const choice = intent.choice
        get.set(intentAtom, { _tag: "Cancelling", choice })
        const shouldCancelDownload = Option.exists(
          Result.value(get(modelsAtom)),
          ({ state }) => state.models.some(({ targetId, download }) =>
            targetId === choice.targetId && download._tag === "Downloading"),
        )
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
          if (shouldCancelDownload) {
            yield* get.setResult(mutations.cancel, {
              payload: { targetId: choice.targetId },
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
          Effect.ensuring(Effect.sync(() =>
            get.set(intentAtom, { _tag: "Idle" }))),
        )
      }

      const choice = command.choice
      get.set(intentAtom, { _tag: "Selecting", choice })
      return Effect.gen(function* () {
        yield* get.setResult(mutations.download, {
          payload: { targetId: choice.targetId },
          reactivityKeys: [
            LocalModelsMirror.id,
            ProviderModelCatalogMirror.id,
          ],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "download",
            message: error.message,
          })),
        )
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
    }),
    [intentAtom, modelsAtom, mutations, slotsAtom],
  )
  const runWorkflow = useAtomSet(workflowAtom)
  const setupAtom = useMemo(
    () => Atom.make((get) => {
      const intent = get(intentAtom)
      return {
        hardware: Result.map(get(hardwareAtom), ({ state }) => state),
        models: Result.map(get(modelsAtom), ({ state }) => state),
        catalog: Result.map(get(catalogAtom), ({ state }) => state),
        slots: Result.map(get(slotsAtom), ({ state }) => state),
        workflowResult: get(workflowAtom),
        submittedChoice: intent._tag === "Idle" ? null : intent.choice,
        cancelling: intent._tag === "Cancelling",
      }
    }),
    [catalogAtom, hardwareAtom, intentAtom, modelsAtom, slotsAtom, workflowAtom],
  )
  const setup = useAtomValue(setupAtom)

  const select = useCallback((choice: OnboardingModelChoice) => {
    if (Result.isWaiting(setup.workflowResult)) return
    runWorkflow({ _tag: "Select", choice })
  }, [runWorkflow, setup.workflowResult])

  const cancel = useCallback(() => {
    runWorkflow({ _tag: "Cancel" })
  }, [runWorkflow])

  return {
    ...setup,
    slotActions,
    select,
    cancel,
  }
}
