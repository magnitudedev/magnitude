import { useCallback, useMemo } from "react"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Cause, Effect, Stream } from "effect"
import { RpcClient } from "@effect/rpc"
import * as Reactivity from "@effect/experimental/Reactivity"
import {
  MagnitudeRpcs,
  type LocalInferenceUsageSelection,
  type LocalModelDownloadProgress,
  type LocalModelDownloadWireEvent,
} from "@magnitudedev/sdk"
import { usePlatform } from "../platform/platform-context"
import { useAgentClient } from "../state/agent-client-context"

const idleDownloadProgressAtom = Atom.make(
  Result.success<LocalModelDownloadProgress | null, never>(null),
)

const isProgress = (event: LocalModelDownloadWireEvent): event is LocalModelDownloadProgress =>
  !("_tag" in event)

const errorMessage = (error: unknown): string => {
  if (typeof error === "object" && error !== null && "reason" in error && typeof error.reason === "string") {
    return error.reason
  }
  return error instanceof Error ? error.message : String(error)
}

const resultError = <A, E>(result: Result.Result<A, E>): string | null =>
  Result.isFailure(result) ? errorMessage(Cause.squash(result.cause)) : null

/**
 * The download stream is only an invalidation channel. Download status remains
 * authoritative in GetLocalModelDownloadProgress and is never copied into a
 * React state value or a client-owned writable atom.
 */
function useLocalInferenceDownloadInvalidation(operationId: string | null): void {
  const platform = usePlatform()
  const subscriptionAtom = useMemo(
    () => Atom.make(
      Effect.gen(function* () {
        if (!operationId) return
        const rpc = yield* RpcClient.make(MagnitudeRpcs)
        yield* rpc.SubscribeLocalModelDownload({ operationId }).pipe(
          Stream.filter(isProgress),
          Stream.tap(() => Reactivity.invalidate(["localInferenceDownload", "localInference"])),
          Stream.runDrain,
        )
      }).pipe(
        Effect.provide(platform.protocolLayer),
        Effect.catchAllCause((cause) =>
          Cause.isInterruptedOnly(cause)
            ? Effect.void
            : Effect.logError(`Local inference download subscription failed: ${Cause.pretty(cause)}`),
        ),
      ),
    ),
    [operationId, platform.protocolLayer],
  )
  useAtomMount(subscriptionAtom)
}

/**
 * Shared local-inference server state and actions.
 *
 * Queries and mutation Results are the only client sources of server facts.
 * The CLI is responsible only for rendering and presentation state.
 */
export function useLocalInferenceState() {
  const client = useAgentClient()
  const snapshotResult = useLocalInferenceSnapshot()

  const configureUsageAtom = useMemo(
    () => client.mutation("ConfigureLocalInferenceUsage"),
    [client],
  )
  const startDownloadAtom = useMemo(
    () => client.mutation("StartLocalModelDownload"),
    [client],
  )
  const cancelDownloadAtom = useMemo(
    () => client.mutation("CancelLocalModelDownload"),
    [client],
  )
  const activateAtom = useMemo(
    () => client.mutation("ActivateLocalModel"),
    [client],
  )
  const completeAtom = useMemo(
    () => client.mutation("CompleteCliModelSetupOnboarding"),
    [client],
  )

  const configureUsageResult = useAtomValue(configureUsageAtom)
  const startDownloadResult = useAtomValue(startDownloadAtom)
  const cancelDownloadResult = useAtomValue(cancelDownloadAtom)
  const activateResult = useAtomValue(activateAtom)
  const completeResult = useAtomValue(completeAtom)

  const runConfigureUsage = useAtomSet(configureUsageAtom, { mode: "promise" })
  const runStartDownload = useAtomSet(startDownloadAtom, { mode: "promise" })
  const runCancelDownload = useAtomSet(cancelDownloadAtom, { mode: "promise" })
  const runActivate = useAtomSet(activateAtom, { mode: "promise" })
  const runComplete = useAtomSet(completeAtom, { mode: "promise" })

  const startedOperationId = Result.isSuccess(startDownloadResult)
    ? startDownloadResult.value.operationId
    : null
  useLocalInferenceDownloadInvalidation(startedOperationId)

  const progressAtom = useMemo(
    () => startedOperationId
      ? client.query("GetLocalModelDownloadProgress", { operationId: startedOperationId }, {
        reactivityKeys: ["localInferenceDownload"],
      })
      : idleDownloadProgressAtom,
    [client, startedOperationId],
  )
  const progressResult = useAtomValue(progressAtom)
  const progress = Result.isSuccess(progressResult) ? progressResult.value : null
  const operationId = progress?.status === "failed" || progress?.status === "cancelled"
    ? null
    : startedOperationId

  const configureUsage = useCallback(async (usage: LocalInferenceUsageSelection) => {
    try {
      return await runConfigureUsage({
        payload: usage,
        reactivityKeys: ["localInference"],
      })
    } catch {
      return undefined
    }
  }, [runConfigureUsage])

  const startDownload = useCallback(async (configurationId: string) => {
    try {
      await runStartDownload({
        payload: { configurationId },
        reactivityKeys: ["localInference", "localInferenceDownload"],
      })
    } catch {
      // The mutation Result is rendered as the authoritative failure state.
    }
  }, [runStartDownload])

  const cancelDownload = useCallback(async () => {
    if (!operationId) return
    try {
      await runCancelDownload({
        payload: { operationId },
        reactivityKeys: ["localInference", "localInferenceDownload"],
      })
    } catch {
      // The mutation Result is rendered as the authoritative failure state.
    }
  }, [operationId, runCancelDownload])

  const activate = useCallback(async (selectionId: string) => {
    try {
      await runActivate({
        payload: { selectionId },
        reactivityKeys: ["localInference", "modelConfig"],
      })
      return true
    } catch {
      return false
    }
  }, [runActivate])

  const completeOnboarding = useCallback(async () => {
    try {
      await runComplete({
        payload: {},
        reactivityKeys: ["localInference", "modelConfig", "apiKey"],
      })
      return true
    } catch {
      return false
    }
  }, [runComplete])

  const error = progress?.status === "failed"
    ? progress.message ?? "The model download failed"
    : [
      resultError(configureUsageResult),
      resultError(startDownloadResult),
      resultError(cancelDownloadResult),
      resultError(activateResult),
      resultError(completeResult),
      resultError(progressResult),
      resultError(snapshotResult),
    ].find((message): message is string => message !== null) ?? null

  return {
    snapshot: Result.isSuccess(snapshotResult) ? snapshotResult.value : null,
    snapshotLoading: Result.isInitial(snapshotResult) || Result.isWaiting(snapshotResult),
    operationId,
    downloadConfigurationId: progress?.selectionId ?? null,
    progress,
    error,
    busy: [
      configureUsageResult,
      startDownloadResult,
      cancelDownloadResult,
      activateResult,
      completeResult,
    ].some(Result.isWaiting),
    configureUsage,
    startDownload,
    cancelDownload,
    activate,
    completeOnboarding,
  }
}

/** Shared authoritative onboarding snapshot query used by every client gate. */
export function useLocalInferenceSnapshot() {
  const client = useAgentClient()
  const snapshotAtom = useMemo(
    () => client.query("GetLocalInferenceOnboardingSnapshot", {}, {
      reactivityKeys: ["localInference", "modelConfig", "apiKey"],
    }),
    [client],
  )
  return useAtomValue(snapshotAtom)
}
