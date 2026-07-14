/**
 * Composer state hook — shared container logic for the composer.
 *
 * Handles: send (with existing session), auto-create session (with dedup),
 * interrupt, bash (returns RunBashResult, writes bashOutputsAtom),
 * slash commands (via app-provided CommandContext), attachment materialization handoff.
 *
 * Both apps use this identically. The only app-specific part is the
 * CommandContext passed in — each app provides its own toast/recent-chats/etc.
 */
import { useCallback, useEffect, useMemo, useRef } from "react"
import { Option } from "effect"
import { useAtomValue, useAtomSet } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { useDisplayState } from "../state/display-state-store"
import { useSlotProfiles } from "./use-slot-profiles"
import { getDraftSessionOwnerId } from "./draft-session-owner"
import { routeSlashCommand, type CommandContext } from "../commands/command-router"
import type { MentionSearchClient } from "./use-file-mentions"
import {
  selectedCwdAtom,
  bashModeAtom,
  settingsOpenAtom,
  usageOpenAtom,
  selectedFilePathAtom,
  pendingUserSubmitAtom,
  sessionActivationPromiseAtom,
  composerTextAtom,
  composerAttachmentsAtom,
  composerHistoryIndexAtom,
  bashOutputsAtom,
  messageHistoryAtom,
  sessionCreateOptionsAtom,
} from "../state/session-atoms"
import { useDisplayViewControllerCore, useSelectedSessionId } from "../display-view-controller/hooks"
import {
  appendMessageToTimeline,
  emptyTimeline,
  INITIAL_ROOT_PAGE_SIZE,
  timelineTail,
  useDisplaySpeculator,
} from "../sync/index"
import type {
  DisplayAttachment,
  MentionAttachment,
  RawMessageAttachment,
} from "@magnitudedev/sdk"
import { createId } from "@magnitudedev/generate-id"
import type { BashResult } from "../utils/bash-executor"

export interface UseComposerStateResult {
  /** Root agent's role label (capitalized) */
  roleLabel: string
  /** Root agent's model display name */
  model: string
  /** Root agent's thinking level (capitalized reasoning effort) */
  thinkingLevel: string
  /** Whether the root agent is currently streaming */
  isStreaming: boolean
  /** Bash mode active flag */
  bashMode: boolean
  /** Toggle bash mode */
  setBashMode: (updater: (prev: boolean) => boolean) => void
  /** Send a message (auto-creates session if none selected). */
  handleSend: (
    text: string,
    attachments?: readonly RawMessageAttachment[],
    opts?: { visibleMessage?: string; taskMode?: boolean },
  ) => void
  /** Interrupt the root agent */
  handleInterrupt: () => void
  /** Run a bash command. Returns the display result and writes to bashOutputsAtom. */
  handleRunBash: (command: string) => Promise<BashResult | null>
  /** Handle a slash command string */
  handleSlashCommand: (cmdText: string) => void
  /** Mention search client (null if runtime not ready) */
  mentionClient: MentionSearchClient | null
  /** Currently selected session ID */
  sessionId: string | null
  /** Currently selected working directory */
  cwd: string | null
}

/**
 * Shared composer state hook.
 * @param commandContext App-specific slash command context (toast, recent chats, etc.)
 */
export function useComposerState(commandContext: CommandContext): UseComposerStateResult {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const displayController = useDisplayViewControllerCore()
  const displaySpeculator = useDisplaySpeculator()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const bashMode = useAtomValue(bashModeAtom)
  const setBashMode = useAtomSet(bashModeAtom)
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const setFilePath = useAtomSet(selectedFilePathAtom)
  const setPendingUserSubmit = useAtomSet(pendingUserSubmitAtom)
  const setComposerText = useAtomSet(composerTextAtom)
  const setComposerAttachments = useAtomSet(composerAttachmentsAtom)
  const setComposerHistoryIndex = useAtomSet(composerHistoryIndexAtom)
  const setSessionActivationPromise = useAtomSet(sessionActivationPromiseAtom)
  const sessionActivationPromise = useAtomValue(sessionActivationPromiseAtom)
  const activationPromiseRef = useRef<Promise<string> | null>(null)
  const activatedSessionIdRef = useRef<string | null>(null)
  const previousSelectedSessionIdRef = useRef<string | null>(selectedSessionId)
  const setBashOutputs = useAtomSet(bashOutputsAtom)
  const setMessageHistory = useAtomSet(messageHistoryAtom)
  const sessionCreateOptions = useAtomValue(sessionCreateOptionsAtom)

  const { rootRoleLabel, rootProfile } = useSlotProfiles()
  const model = rootProfile?.modelDisplayName ?? ""
  const thinkingLevel = rootProfile?.reasoningEffort
    ? rootProfile.reasoningEffort.charAt(0).toUpperCase() + rootProfile.reasoningEffort.slice(1)
    : ""

  const rootActor = useDisplayState((state) => state.actors["root"] ?? null)
  const isStreaming = rootActor?.work.phase === "working"

  useEffect(() => {
    const previous = previousSelectedSessionIdRef.current
    previousSelectedSessionIdRef.current = selectedSessionId
    if (selectedSessionId) {
      activatedSessionIdRef.current = selectedSessionId
    } else if (previous !== null && activationPromiseRef.current === null) {
      activatedSessionIdRef.current = null
    }
  }, [selectedSessionId])

  const sendMutation = useAtomSet(
    client.mutation("SendMessage"),
    { mode: "promise" },
  )
  const createSession = useAtomSet(
    client.mutation("CreateSession"),
    { mode: "promise" },
  )
  const interruptMutation = useAtomSet(client.mutation("Interrupt"))
  const runBashMutation = useAtomSet(
    client.mutation("RunBash"),
    { mode: "promise" },
  )
  const searchMentionsMutation = useAtomSet(
    client.mutation("SearchMentions"),
    { mode: "promise" },
  )
  // Mention client — uses mutation setter, no manual runtime extraction
  const mentionClient = useMemo<MentionSearchClient>(() => ({
    searchMentions(payload: Parameters<MentionSearchClient["searchMentions"]>[0]) {
      return searchMentionsMutation({
        payload: {
          cwd: payload.cwd,
          query: payload.query,
          ...(payload.limit !== undefined ? { limit: payload.limit } : {}),
          ...(payload.visibleLimit !== undefined ? { visibleLimit: payload.visibleLimit } : {}),
          ...(payload.includeRecent !== undefined ? { includeRecent: payload.includeRecent } : {}),
        },
      })
    },
  }), [searchMentionsMutation])

  const toOptimisticDisplayAttachments = (attachments: readonly RawMessageAttachment[]): DisplayAttachment[] =>
    attachments.filter((attachment): attachment is MentionAttachment =>
      attachment.type === "mention_file"
      || attachment.type === "mention_file_range"
      || attachment.type === "mention_directory"
    )

  const handleSend = useCallback((
    text: string,
    attachments?: readonly RawMessageAttachment[],
    opts?: { visibleMessage?: string; taskMode?: boolean },
  ): void => {
    const rawMessageAttachments = attachments ?? []
    const displayAttachments = toOptimisticDisplayAttachments(rawMessageAttachments)
    const taskMode = opts?.taskMode ?? false
    const visibleMessage = opts?.visibleMessage !== undefined ? Option.some(opts.visibleMessage) : Option.none<string>()
    const messageId = createId()
    const displayText = opts?.visibleMessage ?? text
    const activeSessionId = selectedSessionId ?? activatedSessionIdRef.current
    const draftOwnerId = getDraftSessionOwnerId()
    const optimisticOwner = activeSessionId
      ? `send:${messageId}`
      : `activation:${draftOwnerId}:${selectedCwd ?? ""}`

    if (!activeSessionId && !selectedCwd) {
      commandContext.showSystemMessage("Choose a working directory before starting a session.")
      return
    }

    // Add to message history
    setMessageHistory((prev: string[]) => [text, ...prev].slice(0, 50))
    setPendingUserSubmit(true)

    // Optimistic speculator disabled — the real message arrives via stream.
    // const optimistic = displaySpeculator.mutate(
    //   { owner: optimisticOwner, label: "send-message" },
    //   (draft) => {
    //     const cwd = selectedCwd ?? draft.state.session.cwd ?? ""
    //     if (!activeSessionId && !draft.state.session.sessionId) {
    //       draft.state.session = {
    //         sessionId: `draft:${draftOwnerId}`,
    //         title: null,
    //         cwd,
    //       }
    //     }
    //     draft.shape.timelines.root ??= timelineTail(INITIAL_ROOT_PAGE_SIZE)
    //     draft.state.timelines.root ??= emptyTimeline()
    //     draft.state.timelines.root = appendMessageToTimeline(draft.state.timelines.root, {
    //       id: messageId,
    //       type: activeSessionId && isStreaming ? "queued_user_message" : "user_message",
    //       content: displayText,
    //       timestamp: Date.now(),
    //       taskMode,
    //       attachments: displayAttachments,
    //     })
    //   },
    // )
    const optimistic = { remove: () => {} }

    const rollback = (err: unknown): void => {
      const errMsg = err instanceof Error ? err.message : String(err)
      optimistic.remove()
      setPendingUserSubmit(false)
      activationPromiseRef.current = null
      setSessionActivationPromise(null)
      setComposerText(opts?.visibleMessage ?? text)
      setComposerAttachments([])
      setComposerHistoryIndex(-1)
      commandContext.showSystemMessage(`Message failed to send: ${errMsg}`)
    }

    const deliver = async (): Promise<void> => {
      if (activeSessionId) {
        await sendMutation({
          payload: {
            sessionId: activeSessionId,
            messageId: Option.some(messageId),
            content: text,
            taskMode,
            attachments: rawMessageAttachments,
            visibleMessage,
          },
          reactivityKeys: ["sessions"],
        })
        setPendingUserSubmit(false)
        return
      }

      // No session — lazy activation with dedup
      const inFlightActivation = activationPromiseRef.current ?? sessionActivationPromise
      if (inFlightActivation) {
        const sessionId = await inFlightActivation
        await sendMutation({
          payload: {
            sessionId,
            messageId: Option.some(messageId),
            content: text,
            taskMode,
            attachments: rawMessageAttachments,
            visibleMessage,
          },
          reactivityKeys: ["sessions"],
        })
        setPendingUserSubmit(false)
        return
      }

      const promise = createSession({
        payload: {
          cwd: selectedCwd ?? "",
          sessionId: Option.none(),
          initial: Option.some({
            _tag: "message",
            messageId: Option.some(messageId),
            content: text,
            visibleMessage,
            taskMode,
            attachments: rawMessageAttachments,
          }),
          options: sessionCreateOptions,
          draftOwnerId: Option.some(getDraftSessionOwnerId()),
        },
        reactivityKeys: ["sessions"],
      }).then((result) => {
        if (result._tag === "created") {
          activatedSessionIdRef.current = result.metadata.sessionId
          displayController.selectSession(result.metadata.sessionId)
          activationPromiseRef.current = null
          setSessionActivationPromise(null)
          return result.metadata.sessionId
        }
        if (result._tag === "created_message_failed") {
          // Message was sent but promote failed. Select the session — the
          // real message will replace the optimistic one via stream. Do NOT
          // restore text. Show the error.
          activatedSessionIdRef.current = result.sessionId
          displayController.selectSession(result.sessionId)
          activationPromiseRef.current = null
          setSessionActivationPromise(null)
          commandContext.showSystemMessage(`Session created but promotion failed: ${result.error}`)
          return result.sessionId
        }
        // failed — message was not sent. Throw to trigger rollback.
        throw new Error(result.error)
      })
      activationPromiseRef.current = promise
      setSessionActivationPromise(promise)
      await promise
      setPendingUserSubmit(false)
    }

    void deliver().catch(rollback)
  }, [selectedSessionId, selectedCwd, sessionActivationPromise, isStreaming, displaySpeculator, sendMutation, createSession, displayController, setPendingUserSubmit, setComposerText, setComposerAttachments, setComposerHistoryIndex, setSessionActivationPromise, setMessageHistory, sessionCreateOptions, commandContext])

  const handleInterrupt = useCallback(() => {
    if (!selectedSessionId) return
    interruptMutation({
      payload: {
        sessionId: selectedSessionId,
        target: { _tag: "fork", forkId: null },
      },
    })
  }, [selectedSessionId, interruptMutation])

  const handleRunBash = useCallback(async (command: string): Promise<BashResult | null> => {
    if (!selectedSessionId) {
      commandContext.showSystemMessage("Start a session first to run bash commands.")
      return null
    }
    try {
      const result = await runBashMutation({
        payload: {
          sessionId: selectedSessionId,
          command,
        },
      })
      const bashResult: BashResult = {
        id: createId(),
        command,
        stdout: result.stdout,
        stderr: result.stderr,
        exitCode: result.exitCode,
        cwd: result.cwd,
        timestamp: Date.now(),
      }
      // Write to shared bashOutputsAtom — both apps can render bash output
      setBashOutputs((prev: BashResult[]) => [...prev, bashResult])
      return bashResult
    } catch (err) {
      console.error("[Composer] RunBash failed:", err)
      return null
    }
  }, [selectedSessionId, runBashMutation, setBashOutputs, commandContext])

  const handleSlashCommand = useCallback((cmdText: string) => {
    routeSlashCommand(cmdText, commandContext)
  }, [commandContext])

  return {
    roleLabel: rootRoleLabel,
    model,
    thinkingLevel,
    isStreaming,
    bashMode,
    setBashMode,
    handleSend,
    handleInterrupt,
    handleRunBash,
    handleSlashCommand,
    mentionClient,
    sessionId: selectedSessionId,
    cwd: selectedCwd,
  }
}
