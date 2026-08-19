/**
 * Composer state hook — shared container logic for the composer.
 *
 * Handles: send (with existing session), auto-create session (with dedup),
 * interrupt, bash, slash commands (via app-provided CommandContext), and
 * attachment materialization handoff.
 *
 * Both apps use this identically. The only app-specific part is the
 * CommandContext passed in — each app provides its own toast/recent-chats/etc.
 */
import { useCallback, useMemo, useRef } from "react"
import { Effect, Option } from "effect"
import { Atom, useAtomMount, useAtomValue, useAtomSet } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { useDisplayState } from "../state/display-state-store"
import { useSlotProfiles } from "./use-slot-profiles"
import { getDraftSessionOwnerId } from "./draft-session-owner"
import { routeSlashCommand, type CommandContext, type SlashCommandOutcome } from "../commands/command-router"
import type { MentionSearchClient } from "./use-file-mentions"
import {
  selectedCwdAtom,
  selectedProjectIdAtom,
  bashModeAtom,
  settingsOpenAtom,
  usageOpenAtom,
  selectedFilePathAtom,
  pendingUserSubmitAtom,
  composerTextAtom,
  composerAttachmentsAtom,
  composerUploadsAtom,
  composerHistoryIndexAtom,
  messageHistoryAtom,
  sessionCreateOptionsAtom,
} from "../state/session-atoms"
import { useDisplayViewControllerCore, useSelectedSessionId } from "../display-view-controller/hooks"
import {
  presentPendingUserMessage,
  useDisplaySpeculator,
} from "../sync/index"
import type {
  DisplayAttachment,
  RawMessageUpload,
  RawMentionOccurrence,
} from "@magnitudedev/sdk"
import { canonicalExtensionForImageMediaType } from "@magnitudedev/sdk"
import { isRpcOutcomeUnknown } from "@magnitudedev/sdk"
import { createId } from "@magnitudedev/generate-id"
import { formatReasoningEffort } from "../utils/model-properties"
import { isDisplayRootStatusActive } from "../utils/actor-status"

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
    input?: {
      readonly uploads?: readonly RawMessageUpload[]
      readonly mentions?: readonly RawMentionOccurrence[]
    },
    opts?: { visibleMessage?: string; taskMode?: boolean },
  ) => void
  /** Interrupt the root agent */
  handleInterrupt: () => void
  /** Run a bash command. The persisted event is the display source of truth. */
  handleRunBash: (command: string) => Promise<boolean>
  /** Handle a slash command string */
  handleSlashCommand: (cmdText: string) => SlashCommandOutcome
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
  const selectedProjectId = useAtomValue(selectedProjectIdAtom)
  const bashMode = useAtomValue(bashModeAtom)
  const setBashMode = useAtomSet(bashModeAtom)
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const setFilePath = useAtomSet(selectedFilePathAtom)
  const setPendingUserSubmit = useAtomSet(pendingUserSubmitAtom)
  const setComposerText = useAtomSet(composerTextAtom)
  const setComposerAttachments = useAtomSet(composerAttachmentsAtom)
  const composerMentionSegments = useAtomValue(composerAttachmentsAtom)
  const setComposerUploads = useAtomSet(composerUploadsAtom)
  const setComposerHistoryIndex = useAtomSet(composerHistoryIndexAtom)
  const activationPromiseRef = useRef<Promise<string> | null>(null)
  const activatedSessionIdRef = useRef<string | null>(null)
  const previousSelectedSessionIdRef = useRef<string | null>(selectedSessionId)
  const setMessageHistory = useAtomSet(messageHistoryAtom)
  const sessionCreateOptions = useAtomValue(sessionCreateOptionsAtom)

  const { rootRoleLabel, rootProfile } = useSlotProfiles()
  const model = rootProfile?.modelDisplayName ?? ""
  const thinkingLevel = rootProfile?.reasoningEffort
    ? formatReasoningEffort(rootProfile.reasoningEffort)
    : ""

  const rootActor = useDisplayState((state) => state.actors["root"] ?? null)
  const isStreaming = rootActor?.kind === "root" && isDisplayRootStatusActive(rootActor.status)

  const selectedSessionSyncAtom = useMemo(() => Atom.make(Effect.sync(() => {
    const previous = previousSelectedSessionIdRef.current
    previousSelectedSessionIdRef.current = selectedSessionId
    if (selectedSessionId) {
      activatedSessionIdRef.current = selectedSessionId
    } else if (previous !== null && activationPromiseRef.current === null) {
      activatedSessionIdRef.current = null
    }
  })), [selectedSessionId])
  useAtomMount(selectedSessionSyncAtom)

  const sendAtom = useMemo(() => client.rpc.mutation("SendMessage"), [client])
  const createSessionAtom = useMemo(() => client.rpc.mutation("CreateSession"), [client])
  const interruptAtom = useMemo(() => client.rpc.mutation("Interrupt"), [client])
  const runBashAtom = useMemo(() => client.rpc.mutation("RunBash"), [client])
  const searchMentionsAtom = useMemo(() => client.rpc.mutation("SearchMentions"), [client])
  const sendMutation = useAtomSet(sendAtom, { mode: "promise" })
  const createSession = useAtomSet(createSessionAtom, { mode: "promise" })
  const interruptMutation = useAtomSet(interruptAtom)
  const runBashMutation = useAtomSet(runBashAtom, { mode: "promise" })
  const searchMentionsMutation = useAtomSet(searchMentionsAtom, { mode: "promise" })
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

  const handleSend = useCallback((
    text: string,
    input?: {
      readonly uploads?: readonly RawMessageUpload[]
      readonly mentions?: readonly RawMentionOccurrence[]
    },
    opts?: { visibleMessage?: string; taskMode?: boolean },
  ): void => {
    const uploads = input?.uploads ?? []
    const mentions = input?.mentions ?? []
    const taskMode = opts?.taskMode ?? false
    const visibleMessage = opts?.visibleMessage !== undefined ? Option.some(opts.visibleMessage) : Option.none<string>()
    const messageId = createId()
    const displayText = opts?.visibleMessage ?? text
    const activeSessionId = selectedSessionId ?? activatedSessionIdRef.current
    const draftOwnerId = getDraftSessionOwnerId()
    if (!activeSessionId && !selectedCwd) {
      commandContext.showSystemMessage("Choose a working directory before starting a session.")
      return
    }

    // Add to message history
    setMessageHistory((prev: string[]) => [text, ...prev].slice(0, 50))
    setPendingUserSubmit(true)

    const optimistic = presentPendingUserMessage(displaySpeculator, {
      messageId,
      content: displayText,
      taskMode,
      activeSessionId,
      draftSessionId: `draft:${draftOwnerId}`,
      cwd: selectedCwd ?? "",
      attachments: uploads.map((upload): DisplayAttachment => {
        if (upload.type === "raw_text_file") {
          return { type: "mention_file", path: upload.filename }
        }
        const filename = upload.type === "raw_image_file"
          ? upload.filename
          : `clipboard-image.${canonicalExtensionForImageMediaType(upload.mediaType)}`
        return {
          type: "image",
          path: filename,
          filename,
          mediaType: upload.mediaType,
          width: upload.width,
          height: upload.height,
        }
      }),
    })

    const reject = (err: unknown): void => {
      const errMsg = err instanceof Error ? err.message : String(err)
      optimistic.remove()
      setPendingUserSubmit(false)
      activationPromiseRef.current = null
      let restoredText = false
      setComposerText(current => {
        if (current.length > 0) return current
        restoredText = true
        return opts?.visibleMessage ?? text
      })
      if (restoredText) setComposerAttachments([...composerMentionSegments])
      setComposerUploads(current => [...uploads, ...current])
      setComposerHistoryIndex(-1)
      commandContext.showSystemMessage(`Message failed to send: ${errMsg}`)
    }

    const handleFailure = (err: unknown): void => {
      if (isRpcOutcomeUnknown(err)) {
        setPendingUserSubmit(false)
        activationPromiseRef.current = null
        commandContext.showSystemMessage(
          "The connection was lost after sending. Magnitude will keep this message visible while it reconciles with the daemon.",
        )
        return
      }
      reject(err)
    }

    const deliver = async (): Promise<void> => {
      if (activeSessionId) {
        await sendMutation({
          payload: {
            sessionId: activeSessionId,
            messageId: Option.some(messageId),
            content: text,
            taskMode,
            uploads,
            mentions,
            visibleMessage,
          },
          reactivityKeys: ["sessions"],
        })
        setPendingUserSubmit(false)
        return
      }

      // No session — lazy activation with dedup
      const inFlightActivation = activationPromiseRef.current
      if (inFlightActivation) {
        const sessionId = await inFlightActivation
        await sendMutation({
          payload: {
            sessionId,
            messageId: Option.some(messageId),
            content: text,
            taskMode,
            uploads,
            mentions,
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
          projectId: Option.fromNullable(selectedProjectId),
          sessionId: Option.none(),
          initial: Option.some({
            _tag: "message",
            messageId: Option.some(messageId),
            content: text,
            visibleMessage,
            taskMode,
            uploads,
            mentions,
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
          return result.metadata.sessionId
        }
        if (result._tag === "created_message_failed") {
          // Message was sent but promote failed. Select the session — the
          // real message will replace the optimistic one via stream. Do NOT
          // restore text. Show the error.
          activatedSessionIdRef.current = result.sessionId
          displayController.selectSession(result.sessionId)
          activationPromiseRef.current = null
          commandContext.showSystemMessage(`Session created but promotion failed: ${result.error}`)
          return result.sessionId
        }
        // failed — message was not sent. Throw to trigger rollback.
        throw new Error(result.error)
      })
      activationPromiseRef.current = promise
      await promise
      setPendingUserSubmit(false)
    }

    void deliver().catch(handleFailure)
  }, [selectedSessionId, selectedCwd, selectedProjectId, displaySpeculator, sendMutation, createSession, displayController, setPendingUserSubmit, setComposerText, composerMentionSegments, setComposerAttachments, setComposerUploads, setComposerHistoryIndex, setMessageHistory, sessionCreateOptions, commandContext])

  const handleInterrupt = useCallback(() => {
    if (!selectedSessionId) return
    interruptMutation({
      payload: {
        sessionId: selectedSessionId,
        target: { _tag: "fork", forkId: null },
      },
    })
  }, [selectedSessionId, interruptMutation])

  const handleRunBash = useCallback(async (command: string): Promise<boolean> => {
    if (!selectedSessionId) {
      commandContext.showSystemMessage("Start a session first to run bash commands.")
      return false
    }
    try {
      await runBashMutation({
        payload: {
          sessionId: selectedSessionId,
          command,
        },
      })
      return true
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      commandContext.showSystemMessage(`Bash command failed: ${message}`)
      return false
    }
  }, [selectedSessionId, runBashMutation, commandContext])

  const handleSlashCommand = useCallback(
    (cmdText: string): SlashCommandOutcome => routeSlashCommand(cmdText, commandContext),
    [commandContext],
  )

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
