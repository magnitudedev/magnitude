import { useCallback, useMemo } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import { useAgentClient } from "../state/agent-client-context"
import { useLocalInferenceResource } from "./use-reactive-rpc"

export function useLocalInferenceQuery() {
  const snapshot = useLocalInferenceResource()
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
  const mutationBusy = mutationResults.some(Result.isWaiting)
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
    downloadMutation({ payload: { configurationId }, reactivityKeys: ["localInference", "modelCatalog", "modelSlots"] })
  }, [downloadMutation])
  const activateModel = useCallback((selectionId: string): void => {
    activateMutation({ payload: { selectionId }, reactivityKeys: ["localInference", "modelCatalog", "modelSlots"] })
  }, [activateMutation])
  const deleteModel = useCallback((selectionId: string): void => {
    deleteMutation({ payload: { selectionId }, reactivityKeys: ["localInference", "modelCatalog", "modelSlots"] })
  }, [deleteMutation])
  const restart = useCallback((): void => {
    restartMutation({ payload: {}, reactivityKeys: ["localInference", "modelCatalog"] })
  }, [restartMutation])
  const disable = useCallback((): void => {
    disableMutation({ payload: {}, reactivityKeys: ["localInference", "modelCatalog", "modelSlots"] })
  }, [disableMutation])

  return {
    state,
    mutationResults,
    mutationBusy,
    mutationFailure,
    downloadModel,
    activateModel,
    deleteModel,
    restart,
    disable,
  }
}
