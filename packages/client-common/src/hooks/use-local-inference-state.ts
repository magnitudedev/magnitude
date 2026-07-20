import { useCallback, useMemo } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import { createId } from "@magnitudedev/generate-id"
import {
  LocalInferenceMirror,
  ModelCatalogMirror,
  ModelSlotsMirror,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredState } from "./use-mirrored-state"

export function useLocalInferenceQuery() {
  const snapshot = useMirroredState(LocalInferenceMirror)
  return Result.map(snapshot, ({ state }) => state)
}

export function useLocalInferenceState() {
  const client = useAgentClient()
  const state = useLocalInferenceQuery()

  const downloadAtom = useMemo(() => client.mutation("DownloadLocalModel"), [client])
  const activateAtom = useMemo(() => client.mutation("ActivateLocalModel"), [client])
  const deleteAtom = useMemo(() => client.mutation("DeleteLocalModel"), [client])
  const restartAtom = useMemo(() => client.mutation("RestartLocalInference"), [client])
  const disableAtom = useMemo(() => client.mutation("DisableLocalInference"), [client])

  const mutationResults = [
    useAtomValue(downloadAtom),
    useAtomValue(activateAtom),
    useAtomValue(deleteAtom),
    useAtomValue(restartAtom),
    useAtomValue(disableAtom),
  ] as const
  const pending = {
    download: Result.isWaiting(mutationResults[0]),
    activate: Result.isWaiting(mutationResults[1]),
    delete: Result.isWaiting(mutationResults[2]),
    restart: Result.isWaiting(mutationResults[3]),
    disable: Result.isWaiting(mutationResults[4]),
  } as const
  const mutationFailure = mutationResults.reduce(
    (failure, result) => Option.isSome(failure) || !Result.isFailure(result)
      ? failure
      : Option.some(Cause.pretty(result.cause)),
    Option.none<string>(),
  )

  const downloadMutation = useAtomSet(downloadAtom)
  const activateMutation = useAtomSet(activateAtom)
  const deleteMutation = useAtomSet(deleteAtom)
  const restartMutation = useAtomSet(restartAtom)
  const disableMutation = useAtomSet(disableAtom)

  const downloadModel = useCallback((configurationId: string): void => {
    downloadMutation({ payload: { configurationId, requestId: createId() }, reactivityKeys: [LocalInferenceMirror.id] })
  }, [downloadMutation])
  const activateModel = useCallback((selectionId: string): void => {
    activateMutation({ payload: { selectionId, requestId: createId() }, reactivityKeys: [LocalInferenceMirror.id] })
  }, [activateMutation])
  const deleteModel = useCallback((selectionId: string): void => {
    deleteMutation({
      payload: { selectionId },
      reactivityKeys: [LocalInferenceMirror.id, ModelCatalogMirror.id, ModelSlotsMirror.id],
    })
  }, [deleteMutation])
  const restart = useCallback((): void => {
    restartMutation({ payload: { requestId: createId() }, reactivityKeys: [LocalInferenceMirror.id] })
  }, [restartMutation])
  const disable = useCallback((): void => {
    disableMutation({
      payload: {},
      reactivityKeys: [LocalInferenceMirror.id, ModelCatalogMirror.id, ModelSlotsMirror.id],
    })
  }, [disableMutation])

  return {
    state,
    mutationResults,
    pending,
    mutationFailure,
    downloadModel,
    activateModel,
    deleteModel,
    restart,
    disable,
  }
}
