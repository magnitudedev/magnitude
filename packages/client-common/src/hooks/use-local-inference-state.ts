import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue, type Atom as EffectAtom } from "@effect-atom/atom-react"
import { Cause, Option } from "effect"
import {
  IcnHardwareMirror,
  IcnInventoryMirror,
  ModelRecipesMirror,
  ModelCatalogMirror,
  ModelSlotsMirror,
} from "@magnitudedev/sdk"
import { useAgentClient } from "../state/agent-client-context"
import { useMirroredStateAtom } from "./use-mirrored-state"
import { deriveLocalInferenceView } from "../utils/local-inference-view"

type HardwareSnapshot = typeof IcnHardwareMirror.snapshotSchema.Type
type InventorySnapshot = typeof IcnInventoryMirror.snapshotSchema.Type
type RecipesSnapshot = typeof ModelRecipesMirror.snapshotSchema.Type

const requireResultValue = <A, E>(
  observed: Result.Result<A, E>,
): Result.Result<A, E> => Option.match(Option.fromNullable(observed), {
  onNone: () => Result.initial(true),
  onSome: (result) => Result.isSuccess(result)
    ? Option.match(Option.fromNullable(result.value), {
        onNone: () => Result.initial(true),
        onSome: () => result,
      })
    : result,
})

export const makeLocalInferenceQueryAtom = <E>(
  hardware: EffectAtom.Atom<Result.Result<HardwareSnapshot, E>>,
  inventory: EffectAtom.Atom<Result.Result<InventorySnapshot, E>>,
  recipes: EffectAtom.Atom<Result.Result<RecipesSnapshot, E>>,
) => Atom.make((get) => Result.map(Result.all({
  hardware: requireResultValue(get(hardware)),
  inventory: requireResultValue(get(inventory)),
  recipes: requireResultValue(get(recipes)),
}), (snapshots) => deriveLocalInferenceView(
  snapshots.hardware.state,
  snapshots.inventory.state,
  snapshots.recipes.state,
)))

export function useLocalInferenceQuery() {
  const hardware = useMirroredStateAtom(IcnHardwareMirror)
  const inventory = useMirroredStateAtom(IcnInventoryMirror)
  const recipes = useMirroredStateAtom(ModelRecipesMirror)
  const state = useMemo(
    () => makeLocalInferenceQueryAtom(hardware, inventory, recipes),
    [hardware, inventory, recipes],
  )
  return useAtomValue(state)
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
    downloadMutation({ payload: { configurationId }, reactivityKeys: [IcnInventoryMirror.id] })
  }, [downloadMutation])
  const activateModel = useCallback((modelId: string): void => {
    activateMutation({ payload: { modelId }, reactivityKeys: [IcnInventoryMirror.id] })
  }, [activateMutation])
  const deleteModel = useCallback((modelId: string): void => {
    deleteMutation({
      payload: { modelId },
      reactivityKeys: [IcnInventoryMirror.id, ModelCatalogMirror.id, ModelSlotsMirror.id],
    })
  }, [deleteMutation])
  const restart = useCallback((): void => {
    restartMutation({ payload: {}, reactivityKeys: [IcnInventoryMirror.id] })
  }, [restartMutation])
  const disable = useCallback((): void => {
    disableMutation({
      payload: {},
      reactivityKeys: [IcnInventoryMirror.id, ModelCatalogMirror.id, ModelSlotsMirror.id],
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
