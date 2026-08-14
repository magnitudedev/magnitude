import { stripVTControlCharacters } from "node:util"
import { Option } from "effect"
import {
  forkIdToKey,
  type DisplayMessage,
  type DisplayTimelineEntry,
  type DisplayViewSnapshot,
  type ToolStepPresentation,
} from "@magnitudedev/sdk"

export interface HeadlessOutput {
  readonly lines: readonly string[]
  readonly toolCount: number
}

const terminalToolPhases: ReadonlySet<ToolStepPresentation["phase"]> = new Set([
  "completed",
  "error",
  "rejected",
  "interrupted",
])
const dataOnlyMessageTypes: ReadonlySet<DisplayMessage["type"]> = new Set([
  "fork_activity",
  "fork_result",
  "worker_resumed",
  "worker_finished",
  "worker_killed",
  "worker_user_killed",
])
const unsafeTerminalControlCharacters = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g
const unsafeUnicodeDisplayControlCharacters = /[\u061c\u200e\u200f\u2028-\u202e\u2066-\u2069]/g
const defaultIgnorableUnicodeCharacters = /[\u00ad\u034f\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200d\u2060-\u2065\u206a-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\u{1bca0}-\u{1bcaf}\u{1d173}-\u{1d17a}\ufff0-\ufff8\u{e0000}-\u{e0fff}]/gu
const cliOwnedRecordPrefix = /^(?:> |\$ |↳ |▶ |· |✓ |✗ |⚠ |■ |▸ |→ |✎ |\/ Search |◫ |⌕ |↓ |◇ |↶ |◉ |Connection issue: retrying|Session: |Error:)/

export function sanitizeHeadlessText(value: string): string {
  return stripVTControlCharacters(value.replace(/\r\n?/g, "\n"))
    .replace(/\t/g, " ")
    .replace(/\n/g, "\\n")
    .replace(defaultIgnorableUnicodeCharacters, "")
    .replace(
      unsafeUnicodeDisplayControlCharacters,
      (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
    )
    .replace(unsafeTerminalControlCharacters, "")
}

/**
 * Renders only authoritative, materialized display state. Message IDs and tool
 * message IDs make reconnect/resync snapshots idempotent; streaming assistant
 * messages and running tools remain pending until their terminal projection is
 * visible.
 */
export function createHeadlessOutputRenderer() {
  const emittedPresentationMessageIds = new Set<string>()
  const emittedDataMessageIds = new Set<string>()
  const seenToolMessageIds = new Set<string>()
  const countedToolMessageIds = new Set<string>()
  const countedWorkerControlMessageIds = new Set<string>()
  const workerToolProgress = new Map<string, number>()
  const completedWorkerToolCounts = new Map<string, number>()
  const workerCompletionTimestamps = new Map<string, number>()
  const workerResumeGenerations = new Map<string, number>()
  const workerCompletionResumeGenerations = new Map<string, number>()

  const observeWorkerCompletion = (
    message: Extract<DisplayMessage, { readonly type: "worker_finished" }>,
  ): boolean => {
    const previousCompletionTools = completedWorkerToolCounts.get(message.forkId)
    const previousTools = Math.max(
      workerToolProgress.get(message.forkId) ?? 0,
      previousCompletionTools ?? 0,
    )
    const previousTimestamp = workerCompletionTimestamps.get(message.forkId)
    const resumeGeneration = workerResumeGenerations.get(message.workerId) ?? 0
    const previousResumeGeneration = workerCompletionResumeGenerations.get(message.forkId)
    if (message.cumulativeTotalToolsUsed < previousTools) return false
    if (previousCompletionTools !== undefined) {
      if (
        message.cumulativeTotalToolsUsed === previousCompletionTools
        && previousTimestamp !== undefined
        && message.timestamp <= previousTimestamp
        && resumeGeneration === previousResumeGeneration
      ) return false
    }
    completedWorkerToolCounts.set(message.forkId, message.cumulativeTotalToolsUsed)
    workerCompletionResumeGenerations.set(message.forkId, resumeGeneration)
    workerCompletionTimestamps.set(
      message.forkId,
      Math.max(previousTimestamp ?? message.timestamp, message.timestamp),
    )
    return true
  }

  const totalToolCount = (): number => {
    const workerForkIds = new Set([
      ...workerToolProgress.keys(),
      ...completedWorkerToolCounts.keys(),
    ])
    const delegatedWorkerTools = [...workerForkIds].reduce((total, forkId) =>
      total + Math.max(
        workerToolProgress.get(forkId) ?? 0,
        completedWorkerToolCounts.get(forkId) ?? 0,
      ), 0)
    return countedToolMessageIds.size
      + countedWorkerControlMessageIds.size
      + delegatedWorkerTools
  }

  const handleSnapshot = (snapshot: DisplayViewSnapshot): HeadlessOutput => {
    const timeline = rootTimeline(snapshot)
    const messages = timeline?.messages.byId ?? {}
    const messageOrder = timeline?.messages.order ?? []
    const messagePositions = new Map(messageOrder.map((messageId, position) => [messageId, position]))
    const pendingLines: Array<{
      readonly position: number
      readonly timestamp: number
      readonly order: number
      readonly line: string
    }> = []
    const emit = (messageId: string, timestamp: number, line: string): void => {
      pendingLines.push({
        position: messagePositions.get(messageId) ?? messageOrder.length,
        timestamp,
        order: pendingLines.length,
        line: sanitizeHeadlessText(line),
      })
    }
    const isAuthoritativeTool = (messageId: string, toolKey: string): boolean => {
      const message = messages[messageId]
      return messagePositions.has(messageId)
        && message?.type === "tool"
        && message.toolKey === toolKey
    }
    const authoritativeToolFailed = (messageId: string): boolean => {
      const message = messages[messageId]
      return message?.type === "tool"
        && Option.isSome(message.presentation)
        && message.presentation.value.failed
    }

    for (const messageId of messageOrder) {
      const message = messages[messageId]
      if (!message || message.id !== messageId || !dataOnlyMessageTypes.has(message.type)) continue

      const firstObservation = !emittedDataMessageIds.has(messageId)
      let toolProgress: { readonly delta: number; readonly total: number } | null = null
      let staleForkActivity = false
      if (message.type === "fork_activity") {
        const total = totalForkTools(message.toolCounts)
        const activityGeneration = Option.getOrElse(message.resumeCount, () => 0)
        const completedGeneration = workerCompletionResumeGenerations.get(message.forkId)
        staleForkActivity = completedGeneration !== undefined
          && activityGeneration <= completedGeneration
        const previous = Math.max(
          workerToolProgress.get(message.forkId) ?? 0,
          completedWorkerToolCounts.get(message.forkId) ?? 0,
        )
        const resumedInitialActivity = firstObservation && !isInitialForkActivity(message)
        if (resumedInitialActivity) {
          workerToolProgress.set(message.forkId, Math.max(previous, total))
        } else if (!staleForkActivity && total > previous) {
          workerToolProgress.set(message.forkId, total)
          toolProgress = { delta: total - previous, total }
        }
      }

      if (firstObservation) {
        emittedDataMessageIds.add(messageId)
        if (message.type === "worker_resumed") {
          workerResumeGenerations.set(
            message.workerId,
            (workerResumeGenerations.get(message.workerId) ?? 0) + 1,
          )
        }
        if (
          (message.type === "fork_activity" && isInitialForkActivity(message) && !staleForkActivity)
          || message.type === "worker_resumed"
          || message.type === "worker_killed"
        ) {
          countedWorkerControlMessageIds.add(message.id)
        }
        const shouldRender = !staleForkActivity
          && (message.type !== "worker_finished" || observeWorkerCompletion(message))
        const line = shouldRender ? renderDisplayMessage(message) : null
        if (line) emit(message.id, message.timestamp, line)
      }

      if (message.type === "fork_activity" && toolProgress !== null) {
        emit(
          message.id,
          message.timestamp,
          `· [${message.forkId}] +${toolProgress.delta} worker ${toolProgress.delta === 1 ? "tool" : "tools"} · ${toolProgress.total} total`,
        )
      }
    }

    for (const entry of rootEntries(snapshot)) {
      switch (entry.kind) {
        case "message": {
          const message = messages[entry.messageId]
          if (!message || message.id !== entry.messageId || !messagePositions.has(entry.messageId)) continue
          if (emittedPresentationMessageIds.has(entry.messageId)) continue
          if (dataOnlyMessageTypes.has(message.type) && emittedDataMessageIds.has(entry.messageId)) continue
          if (entry.streaming) continue
          emittedPresentationMessageIds.add(entry.messageId)
          const shouldRender = message.type !== "worker_finished" || observeWorkerCompletion(message)
          const line = shouldRender ? renderDisplayMessage(message) : null
          if (line) emit(entry.messageId, entry.timestamp, line)
          break
        }
        case "tool_step": {
          if (!terminalToolPhases.has(entry.step.phase)) continue
          const message = messages[entry.messageId]
          if (!isAuthoritativeTool(entry.messageId, entry.step.toolKey) || message?.type !== "tool") continue
          if (seenToolMessageIds.has(entry.messageId)) continue
          const canonicalStep = Option.getOrNull(message.presentation)
          if (
            canonicalStep === null
            || canonicalStep.toolKey !== message.toolKey
            || !terminalToolPhases.has(canonicalStep.phase)
          ) continue
          seenToolMessageIds.add(entry.messageId)
          countedToolMessageIds.add(entry.messageId)
          emit(
            entry.messageId,
            message.timestamp,
            canonicalStep
              ? renderToolStep(canonicalStep)
              : renderToolCount(message.toolKey, 1, false),
          )
          break
        }
        case "tool_summary": {
          if (!terminalToolPhases.has(entry.summary.phase)) continue
          if (
            entry.messageIds.length === 0
            || new Set(entry.messageIds).size !== entry.messageIds.length
            || entry.summary.count !== entry.messageIds.length
            || !entry.messageIds.every((messageId) =>
              isAuthoritativeTool(messageId, entry.summary.toolKey)
            )
          ) continue
          const canonicalPresentations = entry.messageIds.flatMap((messageId) => {
            const message = messages[messageId]
            return message?.type === "tool" && Option.isSome(message.presentation)
              ? [message.presentation.value]
              : []
          })
          const canonicalSummaryPhase = canonicalPresentations.some((presentation) =>
            presentation.phase !== "completed"
          ) ? "error" : "completed"
          if (
            canonicalPresentations.some((presentation) =>
              presentation.toolKey !== entry.summary.toolKey
              || !terminalToolPhases.has(presentation.phase)
            )
            || entry.summary.phase !== canonicalSummaryPhase
          ) continue
          const spansOtherMessages = messageIdsSpanOtherMessages(entry.messageIds, messagePositions)
          const newMessageIds = entry.messageIds.filter((messageId) =>
            !seenToolMessageIds.has(messageId)
          )
          if (newMessageIds.length === 0) continue
          for (const messageId of newMessageIds) {
            seenToolMessageIds.add(messageId)
            countedToolMessageIds.add(messageId)
          }

          if (newMessageIds.length === entry.messageIds.length && !spansOtherMessages) {
            const failed = entry.messageIds.some((messageId) => {
              const message = messages[messageId]
              return message?.type === "tool"
                && Option.isSome(message.presentation)
                && message.presentation.value.failed
            })
            emit(
              newMessageIds[0] ?? entry.id,
              entry.timestamp,
              renderToolCount(entry.summary.toolKey, newMessageIds.length, failed),
            )
            break
          }

          const newSteps = newMessageIds.flatMap((messageId) => {
            const message = messages[messageId]
            if (message?.type !== "tool" || Option.isNone(message.presentation)) return []
            return terminalToolPhases.has(message.presentation.value.phase)
              ? [{ messageId, step: message.presentation.value }]
              : []
          })
          if (newSteps.length === newMessageIds.length) {
            for (const { messageId, step } of newSteps) {
              emit(messageId, entry.timestamp, renderToolStep(step))
            }
          } else if (spansOtherMessages) {
            for (const messageId of newMessageIds) {
              emit(
                messageId,
                entry.timestamp,
                renderToolCount(
                  entry.summary.toolKey,
                  1,
                  authoritativeToolFailed(messageId),
                  true,
                ),
              )
            }
          } else {
            emit(
              newMessageIds[0] ?? entry.id,
              entry.timestamp,
              renderToolCount(
                entry.summary.toolKey,
                newMessageIds.length,
                newMessageIds.some(authoritativeToolFailed),
                true,
              ),
            )
          }
          break
        }
      }
    }

    const lines = pendingLines
      .sort((a, b) => a.position - b.position || a.timestamp - b.timestamp || a.order - b.order)
      .map(({ line }) => line)
    return { lines, toolCount: totalToolCount() }
  }

  return {
    handleSnapshot,
    getToolCount: totalToolCount,
  }
}

export function renderUsageSummary(elapsedMs: number, totalTools: number, success: boolean): string {
  const label = success ? "Finished" : "Failed"
  const icon = success ? "✓" : "✗"
  return `${icon} ${label} · ${formatDuration(Math.floor(elapsedMs / 1000))} · ${totalTools} ${
    totalTools === 1 ? "tool" : "tools"
  }`
}

export function renderInterruptedSummary(elapsedMs: number, totalTools: number): string {
  return `■ Interrupted · ${formatDuration(Math.floor(elapsedMs / 1000))} · ${totalTools} ${
    totalTools === 1 ? "tool" : "tools"
  }`
}

export function renderErrorMessage(message: string): string {
  return `✗ ${message}`
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`
}

function rootEntries(snapshot: DisplayViewSnapshot): readonly DisplayTimelineEntry[] {
  return rootTimeline(snapshot)?.presentation.entries ?? []
}

function rootTimeline(snapshot: DisplayViewSnapshot) {
  return snapshot.state.timelines[forkIdToKey(null)]
}

function messageIdsSpanOtherMessages(
  messageIds: readonly string[],
  messagePositions: ReadonlyMap<string, number>,
): boolean {
  const positions = messageIds.flatMap((messageId) => {
    const position = messagePositions.get(messageId)
    return position === undefined ? [] : [position]
  })
  if (positions.length < 2) return false
  positions.sort((a, b) => a - b)
  return positions.some((position, index) => index > 0 && position > positions[index - 1]! + 1)
}

function totalForkTools(
  counts: Extract<DisplayMessage, { readonly type: "fork_activity" }>["toolCounts"],
): number {
  return Object.values(counts).reduce((total, count) => total + count, 0)
}

function renderDisplayMessage(message: DisplayMessage): string | null {
  switch (message.type) {
    case "user_message":
      return message.taskMode ? `▸ Task: ${message.content}` : `> ${message.content}`
    case "queued_user_message":
      return `> [queued] ${message.content}`
    case "assistant_message": {
      const text = sanitizeHeadlessText(message.content.trim())
      if (text.length === 0) return null
      return cliOwnedRecordPrefix.test(text) ? `assistant: ${text}` : text
    }
    case "user_bash_command":
      return `$ ${message.command} · exit ${message.exitCode}`
    case "thinking":
    case "tool":
    case "work_summary":
    case "agent_communication":
      return null
    case "status_indicator":
      return message.message
    case "goal_status":
      return message.status === "started"
        ? `▸ Goal: ${Option.getOrElse(message.objective, () => "started")}`
        : `✓ Goal finished${Option.match(message.evidence, {
            onNone: () => "",
            onSome: (evidence) => ` · ${evidence}`,
          })}`
    case "worker_resumed":
      return `▶ [${message.workerId}] (${message.workerRole}) resumed · ${message.title}`
    case "worker_finished":
      return `✓ [${message.workerId}] (${message.workerRole}) done · ${message.cumulativeTotalToolsUsed} ${
        message.cumulativeTotalToolsUsed === 1 ? "tool" : "tools"
      }`
    case "worker_killed":
      return `■ [${message.workerId}] (${message.workerRole}) killed · ${message.title}`
    case "worker_user_killed":
      return `■ [${message.workerId}] (${message.workerRole}) stopped by user`
    case "interrupted":
      return message.context === "root" ? "■ Interrupted" : "■ Worker interrupted"
    case "error":
      return renderErrorMessage(message.message)
    case "fork_result":
      return `✓ Worker result · ${message.task}`
    case "fork_activity":
      return message.status === "running" && isInitialForkActivity(message)
        ? `▶ [${message.forkId}] (${message.role}) started · ${message.name}`
        : null
  }
}

function isInitialForkActivity(
  message: Extract<DisplayMessage, { readonly type: "fork_activity" }>,
): boolean {
  return Option.getOrElse(message.resumeCount, () => 0) === 0
}

function renderToolStep(step: ToolStepPresentation): string {
  if (step.icon === "tool") {
    const rendered = step.label
    return step.failed
      ? `✗ ${rendered}${step.errorText ? ` · ${step.errorText}` : ""}`
      : `· ${rendered}`
  }

  const rendered = (() => {
    switch (step.toolKey) {
      case "shell":
        return `$ ${step.command}${step.exitCode === null ? "" : ` · exit ${step.exitCode}`}`
      case "fileRead":
        return `→ Read ${step.path ?? "(unknown)"}${
          step.lineCount === null ? "" : ` · ${step.lineCount} ${step.lineCount === 1 ? "line" : "lines"}`
        }`
      case "fileWrite":
        return `→ Write ${step.displayPath ?? step.path ?? "(unknown)"} · ${step.lineCount} ${
          step.lineCount === 1 ? "line" : "lines"
        }`
      case "fileEdit":
        return `✎ Edit ${step.displayPath ?? step.path ?? "(unknown)"} · +${step.addedCount} -${step.removedCount}`
      case "fileSearch":
        return `/ Search "${step.pattern ?? ""}" · ${step.matchCount} ${
          step.matchCount === 1 ? "match" : "matches"
        } in ${step.fileCount} ${step.fileCount === 1 ? "file" : "files"}`
      case "fileTree":
        return `◫ List ${step.path} · ${step.fileCount} ${step.fileCount === 1 ? "file" : "files"}, ${
          step.dirCount
        } ${step.dirCount === 1 ? "dir" : "dirs"}`
      case "fileView":
        return `→ View ${step.path ?? "(unknown)"}`
      case "webSearch":
        return `⌕ Search web for "${step.query ?? ""}" · ${step.sourceCount} ${
          step.sourceCount === 1 ? "source" : "sources"
        }`
      case "webFetch":
        return `↓ Fetch ${step.url ?? "(unknown url)"}`
      case "skill":
        return `▸ Activate skill ${step.skillName ?? "(unknown)"}`
      case "checkpointChanges":
      case "checkpointRollback":
        return `${step.isRollback ? "↶ Roll back" : "◇ Checkpoint"} · ${step.fileCount} ${
          step.fileCount === 1 ? "file" : "files"
        }`
      case "spawnWorker":
        return `▶ Start worker ${step.agentId ?? "worker"}${step.title ? ` · ${step.title}` : ""}`
      case "queryImage":
        return `◉ Inspect image ${step.path ?? "(unknown)"}`
    }
  })()

  if (!step.failed) return rendered
  const errorText = "errorText" in step ? step.errorText : null
  return `✗ ${rendered}${errorText ? ` · ${errorText}` : ""}`
}

function renderToolCount(
  toolKey: string,
  count: number,
  failed: boolean,
  delta = false,
): string {
  const rendered = `${toolKey} × ${delta ? "+" : ""}${count}`
  return failed ? `✗ ${rendered}` : `· ${rendered}`
}
