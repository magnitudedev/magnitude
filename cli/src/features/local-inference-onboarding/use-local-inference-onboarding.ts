import { useCallback, useMemo, useState } from "react"
import { Atom, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Effect, Stream } from "effect"
import { RpcClient } from "@effect/rpc"
import * as Reactivity from "@effect/experimental/Reactivity"
import {
  MagnitudeRpcs,
  type LocalInferenceOnboardingSnapshot,
  type LocalInferenceUsageSelection,
  type LocalModelDownloadProgress,
  type LocalModelDownloadWireEvent,
} from "@magnitudedev/sdk"
import { useAgentClient, usePlatform, useSettingsState } from "@magnitudedev/client-common"
import { authSourceAtom } from "../../state/cli-atoms"

const errorMessage = (error: unknown): string => {
  if (typeof error === "object" && error !== null && "reason" in error && typeof error.reason === "string") {
    return error.reason
  }
  return error instanceof Error ? error.message : String(error)
}

const isProgress = (event: LocalModelDownloadWireEvent): event is LocalModelDownloadProgress =>
  !("_tag" in event)

export function useLocalInferenceOnboarding() {
  const client = useAgentClient()
  const platform = usePlatform()
  const settings = useSettingsState()
  const authSource = useAtomValue(authSourceAtom)
  const [operationId, setOperationId] = useState<string | null>(null)
  const [downloadConfigurationId, setDownloadConfigurationId] = useState<string | null>(null)
  const [progress, setProgress] = useState<LocalModelDownloadProgress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [usageSnapshot, setUsageSnapshot] = useState<LocalInferenceOnboardingSnapshot | null>(null)

  const configureUsageMutation = useAtomSet(client.mutation("ConfigureLocalInferenceUsage"), { mode: "promise" })
  const startDownloadMutation = useAtomSet(client.mutation("StartLocalModelDownload"), { mode: "promise" })
  const cancelDownloadMutation = useAtomSet(client.mutation("CancelLocalModelDownload"), { mode: "promise" })
  const activateMutation = useAtomSet(client.mutation("ActivateLocalModel"), { mode: "promise" })
  const completeMutation = useAtomSet(client.mutation("CompleteCliModelSetupOnboarding"), { mode: "promise" })

  const progressAtom = useMemo(
    () => Atom.make(
      Effect.gen(function* () {
        if (!operationId) return
        const rpc = yield* RpcClient.make(MagnitudeRpcs)
        yield* rpc.SubscribeLocalModelDownload({ operationId }).pipe(
          Stream.filter(isProgress),
          Stream.tap((event) => Effect.sync(() => {
            setProgress(event)
            if (event.status === "failed") setError(event.message ?? "The model download failed")
            if (event.status === "failed" || event.status === "cancelled") {
              setOperationId(null)
              setDownloadConfigurationId(null)
            }
          })),
          Stream.tap((event) =>
            event.status === "ready" || event.status === "cancelled" || event.status === "failed"
              ? Reactivity.invalidate(["localInference"])
              : Effect.void,
          ),
          Stream.runDrain,
        )
      }).pipe(
        Effect.provide(platform.protocolLayer),
        Effect.catchAllCause((cause) =>
          Cause.isInterruptedOnly(cause)
            ? Effect.void
            : Effect.sync(() => setError(`Download progress disconnected: ${Cause.pretty(cause)}`)),
        ),
      ),
    ),
    [operationId, platform.protocolLayer],
  )
  useAtomMount(progressAtom)

  const run = useCallback(async <A,>(effect: () => Promise<A>): Promise<A | undefined> => {
    if (busy) return undefined
    setBusy(true)
    setError(null)
    try {
      return await effect()
    } catch (cause) {
      setError(errorMessage(cause))
      return undefined
    } finally {
      setBusy(false)
    }
  }, [busy])

  const startDownload = useCallback(async (configurationId: string) => {
    const result = await run(() => startDownloadMutation({
      payload: { configurationId },
      reactivityKeys: ["localInference"],
    }))
    if (result) {
      setOperationId(result.operationId)
      setDownloadConfigurationId(configurationId)
      setProgress({
        operationId: result.operationId,
        status: "queued",
        completedBytes: 0,
        totalBytes: 0,
        resumable: true,
        selectionId: configurationId,
      })
    }
  }, [run, startDownloadMutation])

  const cancelDownload = useCallback(async () => {
    if (!operationId) return
    const result = await run(() => cancelDownloadMutation({
      payload: { operationId },
      reactivityKeys: ["localInference"],
    }))
    if (result) {
      setOperationId(null)
      setDownloadConfigurationId(null)
      setProgress(null)
    }
  }, [cancelDownloadMutation, operationId, run])

  const activate = useCallback(async (selectionId: string) => {
    const result = await run(() => activateMutation({
      payload: { selectionId },
      reactivityKeys: ["localInference", "modelConfig"],
    }))
    return result !== undefined
  }, [activateMutation, run])

  const configureUsage = useCallback(async (usage: LocalInferenceUsageSelection) => {
    const result = await run(() => configureUsageMutation({
      payload: usage,
      reactivityKeys: ["localInference"],
    }))
    if (result) setUsageSnapshot(result)
    return result
  }, [configureUsageMutation, run])

  const configureCloud = useCallback(async (key: string) => {
    await settings.saveApiKey(key)
  }, [settings.saveApiKey])

  const completeOnboarding = useCallback(async () => {
    await completeMutation({
      payload: {},
      reactivityKeys: ["localInference", "modelConfig", "apiKey"],
    })
    return true
  }, [completeMutation])

  return {
    operationId,
    downloadConfigurationId,
    progress,
    error,
    busy,
    usageSnapshot,
    configureUsage,
    startDownload,
    cancelDownload,
    activate,
    configureCloud,
    completeOnboarding,
    cloudKeyAlreadySet: settings.keyAlreadySet || authSource.source === "env",
  }
}
