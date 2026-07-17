import { useCallback, useMemo } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import type { LocalInferenceUsageSelection } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useLocalInferenceResource } from "./use-reactive-rpc"

export function useLocalInferenceQuery() {
  const snapshot = useLocalInferenceResource()
  return Result.map(snapshot, ({ state }) => state)
}

export function useLocalInferenceState() {
  const client = useAgentClient()
  const state = useLocalInferenceQuery()

  const configureAtom = useMemo(() => client.mutation("ConfigureLocalInferenceUsage"), [client])
  const installAtom = useMemo(() => client.mutation("InstallManagedLlamaCpp"), [client])
  const refreshInstallationsAtom = useMemo(() => client.mutation("RefreshLocalInferenceInstallations"), [client])
  const downloadAtom = useMemo(() => client.mutation("DownloadLocalModel"), [client])
  const activateAtom = useMemo(() => client.mutation("ActivateLocalModel"), [client])
  const deleteAtom = useMemo(() => client.mutation("DeleteLocalModel"), [client])
  const restartAtom = useMemo(() => client.mutation("RestartLocalInference"), [client])
  const disableAtom = useMemo(() => client.mutation("DisableLocalInference"), [client])

  const mutationResults = [
    useAtomValue(configureAtom),
    useAtomValue(installAtom),
    useAtomValue(refreshInstallationsAtom),
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

  const configureMutation = useAtomSet(configureAtom)
  const installMutation = useAtomSet(installAtom)
  const refreshInstallationsMutation = useAtomSet(refreshInstallationsAtom)
  const downloadMutation = useAtomSet(downloadAtom)
  const activateMutation = useAtomSet(activateAtom)
  const deleteMutation = useAtomSet(deleteAtom)
  const restartMutation = useAtomSet(restartAtom)
  const disableMutation = useAtomSet(disableAtom)

  const configureUsage = useCallback((selection: LocalInferenceUsageSelection): void => {
    configureMutation({ payload: selection, reactivityKeys: ["localInference", "modelSlots"] })
  }, [configureMutation])
  const installLlamaCpp = useCallback((): void => {
    installMutation({ payload: {}, reactivityKeys: ["localInference", "modelCatalog", "modelSlots"] })
  }, [installMutation])
  const refreshInstallations = useCallback((): void => {
    refreshInstallationsMutation({ payload: {}, reactivityKeys: ["localInference", "modelCatalog", "modelSlots"] })
  }, [refreshInstallationsMutation])
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
    configureUsage,
    installLlamaCpp,
    refreshInstallations,
    downloadModel,
    activateModel,
    deleteModel,
    restart,
    disable,
  }
}
