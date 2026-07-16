import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Effect, Stream } from "effect"
import { RpcClient } from "@effect/rpc"
import { MagnitudeRpcs, type LocalInferenceError, type LocalInferenceState, type LocalInferenceUsageSelection } from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { usePlatform } from "../platform/platform-context"

const localInferenceStreamAtom = Atom.make<Result.Result<LocalInferenceState, LocalInferenceError>>(Result.initial())

export function useLocalInferenceQuery() {
  const platform = usePlatform()
  const setState = useAtomSet(localInferenceStreamAtom)
  const subscriptionAtom = useMemo(() => Atom.make(Effect.gen(function* () {
    const client = yield* RpcClient.make(MagnitudeRpcs)
    yield* client.StreamLocalInferenceState({}).pipe(
      Stream.runForEach((state) => Effect.sync(() => setState(Result.success(state)))),
      Effect.catchAllCause((cause) => Effect.logError(`StreamLocalInferenceState error: ${Cause.pretty(cause)}`)),
      Effect.forkScoped,
    )
  }).pipe(Effect.provide(platform.protocolLayer))), [platform.protocolLayer, setState])
  useAtomMount(subscriptionAtom)
  return useAtomValue(localInferenceStreamAtom)
}

export function useLocalInferenceState() {
  const client = useAgentClient()
  const state = useLocalInferenceQuery()

  const configureAtom = useMemo(() => client.mutation("ConfigureLocalInferenceUsage"), [client])
  const installAtom = useMemo(() => client.mutation("InstallLocalInferenceDistribution"), [client])
  const downloadAtom = useMemo(() => client.mutation("DownloadLocalModel"), [client])
  const activateAtom = useMemo(() => client.mutation("ActivateLocalModel"), [client])
  const deleteAtom = useMemo(() => client.mutation("DeleteLocalModel"), [client])
  const restartAtom = useMemo(() => client.mutation("RestartLocalInference"), [client])
  const disableAtom = useMemo(() => client.mutation("DisableLocalInference"), [client])

  const mutationResults = [
    useAtomValue(configureAtom),
    useAtomValue(installAtom),
    useAtomValue(downloadAtom),
    useAtomValue(activateAtom),
    useAtomValue(deleteAtom),
    useAtomValue(restartAtom),
    useAtomValue(disableAtom),
  ] as const

  const configureMutation = useAtomSet(configureAtom)
  const installMutation = useAtomSet(installAtom)
  const downloadMutation = useAtomSet(downloadAtom)
  const activateMutation = useAtomSet(activateAtom)
  const deleteMutation = useAtomSet(deleteAtom)
  const restartMutation = useAtomSet(restartAtom)
  const disableMutation = useAtomSet(disableAtom)

  const configureUsage = useCallback((selection: LocalInferenceUsageSelection): void => {
    configureMutation({ payload: selection, reactivityKeys: ["localInference"] })
  }, [configureMutation])
  const installDistribution = useCallback((): void => {
    installMutation({ payload: {}, reactivityKeys: ["localInference"] })
  }, [installMutation])
  const downloadModel = useCallback((configurationId: string): void => {
    downloadMutation({ payload: { configurationId }, reactivityKeys: ["localInference"] })
  }, [downloadMutation])
  const activateModel = useCallback((selectionId: string): void => {
    activateMutation({ payload: { selectionId }, reactivityKeys: ["localInference", "modelCatalog", "modelSlots"] })
  }, [activateMutation])
  const deleteModel = useCallback((selectionId: string): void => {
    deleteMutation({ payload: { selectionId }, reactivityKeys: ["localInference"] })
  }, [deleteMutation])
  const restart = useCallback((): void => {
    restartMutation({ payload: {}, reactivityKeys: ["localInference"] })
  }, [restartMutation])
  const disable = useCallback((): void => {
    disableMutation({ payload: {}, reactivityKeys: ["localInference", "modelCatalog", "modelSlots"] })
  }, [disableMutation])

  return {
    state,
    mutationResults,
    configureUsage,
    installDistribution,
    downloadModel,
    activateModel,
    deleteModel,
    restart,
    disable,
  }
}
