import { useCallback, useMemo, useState } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Option, Schema } from "effect"
import {
  LocalModelsMirror,
  ModelSlotsMirror,
  OnboardingMirror,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogMirror,
  type LocalModelsState,
  type ModelOfferingTargetId,
  type ProviderModelId,
  type ReasoningEffort,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import {
  useLocalModels,
  useModelSlotActions,
  useModelSlots,
  useProviderModelCatalog,
  useLocalInferenceHardware,
} from "./use-local-inference-state"

export interface OnboardingModelChoice {
  readonly targetId: ModelOfferingTargetId
  readonly providerModelId: ProviderModelId
  readonly reasoningEffort: ReasoningEffort
}

interface OnboardingModelWorkflowInput {
  readonly choice: OnboardingModelChoice
  readonly downloadRequired: boolean
}

export const onboardingModelDownloadRequired = (
  models: LocalModelsState,
  targetId: ModelOfferingTargetId,
): Option.Option<boolean> => {
  const model = models.models.find((candidate) => candidate.targetId === targetId)
    ?? (models.recommendations._tag === "Ready"
      ? models.recommendations.catalog.find((candidate) =>
          candidate.targetId === targetId)
      : undefined)
  return Option.map(
    Option.fromNullable(model),
    ({ download }) => download._tag !== "Downloaded",
  )
}

export class OnboardingModelCommandFailed extends Schema.TaggedError<
  OnboardingModelCommandFailed
>()("OnboardingModelCommandFailed", {
  command: Schema.Literal("download", "assign", "load", "complete"),
  message: Schema.String,
}) {}

export const useOnboardingModelSetup = () => {
  const client = useAgentClient()
  const hardware = useLocalInferenceHardware()
  const models = useLocalModels()
  const catalog = useProviderModelCatalog()
  const slots = useModelSlots()
  const slotActions = useModelSlotActions()
  const downloadAtom = client.mutation("DownloadModel")
  const assignAtom = client.mutation("AssignSlot")
  const loadAtom = client.mutation("LoadModel")
  const completeAtom = client.mutation("UpdateOnboardingState")
  const cancelDownloadAtom = client.mutation("CancelModelDownload")
  const cancelDownloadResult = useAtomValue(cancelDownloadAtom)
  const cancelDownload = useAtomSet(cancelDownloadAtom)
  const [submittedChoice, setSubmittedChoice] =
    useState<OnboardingModelChoice | null>(null)

  const workflowAtom = useMemo(
    () => Atom.fn<OnboardingModelWorkflowInput>()(({ choice, downloadRequired }, get) =>
      Effect.gen(function* () {
        if (downloadRequired) {
          yield* get.setResult(downloadAtom, {
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
        }
        yield* get.setResult(assignAtom, {
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
        const load = yield* get.setResult(loadAtom, {
          payload: { slotId: PRIMARY_SLOT_ID },
          reactivityKeys: [ModelSlotsMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "load",
            message: error.message,
          })),
        )
        if (load._tag === "Cancelled") return load
        yield* get.setResult(completeAtom, {
          payload: { completed: true },
          reactivityKeys: [OnboardingMirror.id],
        }).pipe(
          Effect.mapError((error) => new OnboardingModelCommandFailed({
            command: "complete",
            message: error.message,
          })),
        )
        return load
      })),
    [assignAtom, completeAtom, downloadAtom, loadAtom],
  )
  const workflowResult = useAtomValue(workflowAtom)
  const runWorkflow = useAtomSet(workflowAtom)

  const select = useCallback((choice: OnboardingModelChoice) => {
    if (Result.isWaiting(workflowResult)) return
    Result.match(models, {
      onInitial: () => undefined,
      onFailure: () => undefined,
      onSuccess: ({ value }) => Option.match(
        onboardingModelDownloadRequired(value, choice.targetId),
        {
          onNone: () => undefined,
          onSome: (downloadRequired) => {
            setSubmittedChoice(choice)
            runWorkflow({ choice, downloadRequired })
          },
        },
      ),
    })
  }, [models, runWorkflow, workflowResult])

  const cancel = useCallback(() => {
    runWorkflow(Atom.Interrupt)
    const choice = submittedChoice
    setSubmittedChoice(null)
    if (choice === null) return

    Result.match(models, {
      onInitial: () => undefined,
      onFailure: () => undefined,
      onSuccess: ({ value }) => {
        const candidate = value.models.find(({ targetId }) =>
          targetId === choice.targetId)
          ?? (value.recommendations._tag === "Ready"
            ? value.recommendations.catalog.find(({ targetId }) =>
                targetId === choice.targetId)
            : undefined)
        if (candidate?.download._tag === "Downloading"
          || candidate?.preparation._tag === "Calibrating") {
          cancelDownload({
            payload: { targetId: choice.targetId },
            reactivityKeys: [LocalModelsMirror.id],
          })
        }
      },
    })

    Result.match(slots, {
      onInitial: () => undefined,
      onFailure: () => undefined,
      onSuccess: ({ value }) => {
        const primary = value.slots.primary
        if (primary._tag === "ConfiguredLocal"
          && primary.selection.providerId === "local"
          && primary.selection.providerModelId === choice.providerModelId) {
          slotActions.clear(PRIMARY_SLOT_ID)
        }
      },
    })
  }, [
    cancelDownload,
    models,
    runWorkflow,
    slotActions,
    slots,
    submittedChoice,
  ])

  return {
    hardware,
    models,
    catalog,
    slots,
    cancelDownloadResult,
    slotActions,
    workflowResult,
    submittedChoice,
    select,
    cancel,
  }
}
