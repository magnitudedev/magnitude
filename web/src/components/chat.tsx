import { submitsChatMessage } from "./chat/keyboard"
import { useCallback, useContext, useRef, useState } from "react"
import {
  Atom,
  RegistryContext,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Cause, Effect, Option, Schema } from "effect"
import {
  type ChatAssistantMessage,
  chatMessages,
  generateSessionId,
  liveResponse,
  type MessageId,
  ReasoningEffortSchema,
  RequestedResponseSettings,
  ServingModelId,
  type SessionId,
  type WatchedChat,
} from "@magnitudedev/sdk"
import {
  chatComposerText,
  chatDraft,
  chatRequest,
  chatRequestSettings,
  chatRetry,
  chatSend,
  completeChatSubmission,
  defaultResponseSettings,
  reasoningEffortFor,
  selectedChat,
  type ChatRequestSettings,
  localModelProviderModelId,
  localModelServingProfile,
  localModelResidency,
  useLocalModels,
  useAgentClient,
} from "@magnitudedev/client-common"
import {
  ArrowUpIcon,
  ClockCounterClockwiseIcon,
  PlusIcon,
  SlidersHorizontalIcon,
  XIcon,
  SquareIcon,
  CircleNotchIcon,
} from "@phosphor-icons/react"
import { Button } from "./ui/button"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "./ui/select"
import { MagnitudeMark } from "./magnitude-mark"
import { NamedLogo, labForName } from "./model-logo"
import { ErrorNotice } from "./error-notice"
import { useNarrowViewport } from "../lib/viewport"
import { Dialog, DialogContent, DialogTitle } from "./ui/dialog"
import {
  IconButton,
  errorText,
  none,
  number,
  value,
} from "./chat/shared"
import { ChatResponse } from "./chat/response"
import { ChatRequestStatus } from "./chat/activity"
import { ChatSettingsPanel } from "./chat/settings"
import { ChatHistoryPanel } from "./chat/history"
import { ChatThinkingSelect } from "./chat/thinking"

export function ChatPage() {
  const id = useAtomValue(selectedChat)
  const select = useAtomSet(selectedChat)
  return <ChatView key={id ?? "new"} id={id} select={select} />
}
function ChatView({
  id,
  select,
}: {
  id: SessionId | null
  select: (id: SessionId | null) => void
}) {
  const client = useAgentClient()
  const registry = useContext(RegistryContext)
  const clearStart = useAtomSet(client.Chat.StartChat)
  const clearSend = useAtomSet(client.Chat.SendChatMessage)
  const clearRetry = useAtomSet(client.Chat.RetryChatResponse)
  const clearStop = useAtomSet(client.Chat.StopChatResponse)
  const clearError = () => {
    clearStart(Atom.Reset)
    clearSend(Atom.Reset)
    clearRetry(Atom.Reset)
    clearStop(Atom.Reset)
  }
  const [panel, setPanel] = useState<"settings" | "history" | null>(null)
  const draftAtom = chatDraft(id)
  const draft = useAtomValue(draftAtom)
  const setDraft = useAtomSet(draftAtom)
  const setNewDraft = useAtomSet(chatDraft(null))
  // A new draft has no chat yet; an inert atom stands in, so no request is made.
  const watched = useAtomValue(
    id ? client.Chat.WatchChat({ chatId: id }) : unwatched
  ).result
  const chat =
    Result.isSuccess(watched) && watched.value._tag === "open"
      ? watched.value
      : null
  // A chat found absent reads as one removed while watched.
  const removed =
    (Result.isSuccess(watched) && watched.value._tag === "removed") ||
    (Result.isFailure(watched) &&
      Option.exists(
        Cause.failureOption(watched.cause),
        (error) => error._tag === "ChatNotFound"
      ))
  const messages = chat ? chatMessages(chat) : []
  const modelResult = useAtomValue(client.Models.ListServableModels({})).result
  const models = Result.isSuccess(modelResult)
    ? modelResult.value.data.filter(
        (m) =>
          m.architecture.input_modalities.includes("text") &&
          m.architecture.output_modalities.includes("text")
      )
    : []
  const settings: ChatRequestSettings = {
    ...chatRequestSettings(chat?.messages ?? []),
    ...draft.settings,
  }
  const local = useLocalModels()
  const localFor = (id: string) =>
    Result.isSuccess(local)
      ? local.value.models.find((m) =>
          Option.contains(localModelProviderModelId(m), id)
        )
      : undefined
  const contextFor = (m: (typeof models)[number]) => {
    const local = localFor(m.id)
    return local
      ? Option.getOrElse(
          Option.map(localModelServingProfile(local), (p) => p.contextLength),
          () => m.context_length
        )
      : m.context_length
  }
  const modelId = value(settings.model) ?? models[0]?.id
  const model = models.find((m) => m.id === modelId)
  const reasoning = model ? Option.getOrNull(model.reasoning) : null
  const effort = reasoningEffortFor(model, settings.reasoningEffort)
  const effectiveSettings: ChatRequestSettings = {
    ...settings,
    model: modelId ? Option.some(ServingModelId.make(modelId)) : none(),
    reasoningEffort: effort,
    // With no explicit output limit, generation may use the remaining context.
    maxOutputTokens: Option.orElse(settings.maxOutputTokens, () =>
      model ? Option.some(contextFor(model)) : none()
    ),
  }
  const request = chatRequest(effectiveSettings)
  const settingsValid = Schema.is(Schema.typeSchema(RequestedResponseSettings))(
    effectiveSettings
  )
  const patch = (change: Partial<ChatRequestSettings>) =>
    setDraft({ ...draft, settings: { ...draft.settings, ...change } })
  const start = useAtomSet(client.Chat.StartChat, { mode: "promise" })
  const send = useAtomSet(client.Chat.SendChatMessage, { mode: "promise" })
  const retry = useAtomSet(client.Chat.RetryChatResponse, {
    mode: "promise",
  })
  const stop = useAtomSet(client.Chat.StopChatResponse)
  const startState = useAtomValue(client.Chat.StartChat)
  const sendState = useAtomValue(client.Chat.SendChatMessage)
  const retryState = useAtomValue(client.Chat.RetryChatResponse)
  const stopState = useAtomValue(client.Chat.StopChatResponse)
  const pending = startState.waiting || sendState.waiting || retryState.waiting
  const active = messages.find(
    (m): m is ChatAssistantMessage =>
      m._tag === "assistant" && Option.isNone(m.end)
  )
  const input = useRef<HTMLTextAreaElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const follow = useRef(true)
  const lastScrollTop = useRef(0)
  // Following the bottom ends the moment the reader scrolls up and resumes once they return to it;
  // while following, any growth of the transcript (streamed text, an opened thinking panel) is
  // pinned to the bottom before it paints.
  const followScroll = useCallback(() => {
    const node = scroller.current
    if (!node) return
    const distance = node.scrollHeight - node.clientHeight - node.scrollTop
    if (node.scrollTop < lastScrollTop.current && distance > 1) follow.current = false
    else if (distance < 32) follow.current = true
    lastScrollTop.current = node.scrollTop
  }, [])
  const pinTranscript = useCallback((content: HTMLDivElement) => {
    const observer = new ResizeObserver(() => {
      const node = scroller.current
      if (node && follow.current) node.scrollTop = node.scrollHeight
    })
    observer.observe(content)
    return () => observer.disconnect()
  }, [])
  const [notice, setNotice] = useState<string | null>(null)
  const last = messages.at(-1)
  const composerText = chatComposerText(
    draft,
    last?._tag === "assistant" ? last : undefined
  )
  // The one rule for what may be submitted, which the send button and Enter share: a chosen,
  // served model, valid settings, nothing running or being sent, a chat that still exists, and for
  // a new message, its text.
  const submittable = (replaces?: MessageId) =>
    Option.filter(
      request,
      () =>
        !active &&
        !pending &&
        settingsValid &&
        model !== undefined &&
        !(id && !chat) &&
        (replaces !== undefined || composerText.trim() !== "")
    )
  const canSend = Option.isSome(submittable())
  const submit = async (replaces?: MessageId) => {
    const ready = submittable(replaces)
    if (Option.isNone(ready)) return
    const asked = messages.at(-2)
    const text = replaces
      ? asked?._tag === "user"
        ? asked.text
        : ""
      : composerText
    follow.current = true
    try {
      // A new chat starts with its first message; an existing one is sent the next.
      let submitted: { readonly chatId: SessionId; readonly responseId: MessageId; readonly text: string }
      if (!id) {
        const payload = Effect.runSync(
          chatSend(Effect.runSync(generateSessionId), text, ready.value)
        )
        await start(payload)
        submitted = payload
      } else if (replaces) {
        const payload = Effect.runSync(chatRetry(id, replaces, ready.value))
        await retry(payload)
        submitted = { chatId: id, responseId: payload.responseId, text }
      } else {
        const payload = Effect.runSync(chatSend(id, text, ready.value))
        await send(payload)
        submitted = { chatId: id, responseId: payload.responseId, text }
      }
      const latest = registry.get(draftAtom)
      const next = completeChatSubmission(
        latest,
        submitted.text,
        submitted.responseId,
        !!replaces
      )
      setDraft(next)
      if (!id) {
        registry.set(chatDraft(submitted.chatId), next)
        setNewDraft({ ...latest, text: "", submitted: null })
        if (registry.get(selectedChat) === id) select(submitted.chatId)
      }
      input.current?.focus()
    } catch {
      /* The mutation's typed failure is rendered below; the draft stays intact. */
    }
  }
  const error = pending
    ? null
    : errorText(startState) ??
      errorText(sendState) ??
      errorText(retryState) ??
      errorText(stopState)
  const narrow = useNarrowViewport()
  const PanelTitle = narrow ? DialogTitle : "h2"
  const body = (
    <>
      <div className="flex items-center justify-between">
        <PanelTitle className="font-sans text-sm font-semibold">
          {panel === "settings" ? "Chat settings" : "History"}
        </PanelTitle>
        <IconButton label="Close panel" onClick={() => setPanel(null)}>
          <XIcon />
        </IconButton>
      </div>
      {panel === "settings" ? (
        <ChatSettingsPanel
          settings={effectiveSettings}
          patch={patch}
          context={model ? contextFor(model) : undefined}
          reset={() =>
            patch({
              ...defaultResponseSettings,
              model: effectiveSettings.model,
              reasoningEffort: reasoning
                ? Option.some(ReasoningEffortSchema.make(reasoning.default_effort))
                : none(),
            })
          }
        />
      ) : (
        <ChatHistoryPanel
          selected={id}
          open={(next) => {
            select(next)
            setPanel(null)
          }}
        />
      )}
    </>
  )
  return (
    <section aria-label="Chat" className="flex min-h-0 flex-1 flex-col">
      <div className="mb-5 flex shrink-0 items-center justify-between gap-3">
        <h1 className="font-heading text-[24px] font-semibold tracking-tight md:text-[28px]">
          Chat
        </h1>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            disabled={pending}
            onClick={() => {
              clearError()
              select(null)
              setNewDraft({
                text: "",
                settings: effectiveSettings,
                submitted: null,
              })
              input.current?.focus()
            }}
          >
            <PlusIcon />
            New chat
          </Button>
          <IconButton
            label="History"
            pressed={panel === "history"}
            onClick={() => setPanel(panel === "history" ? null : "history")}
          >
            <ClockCounterClockwiseIcon />
          </IconButton>
          <IconButton
            label="Chat settings"
            pressed={panel === "settings"}
            onClick={() => setPanel(panel === "settings" ? null : "settings")}
          >
            <SlidersHorizontalIcon />
          </IconButton>
        </div>
      </div>
      <div className="relative flex min-h-0 flex-1 gap-7">
        <div className="flex min-w-0 flex-1 flex-col">
          <div
            ref={scroller}
            onScroll={followScroll}
            className="chat-scroll min-h-0 flex-1 overflow-y-auto"
          >
            {id && Result.isInitial(watched) ? (
              <div
                role="status"
                className="py-10 text-center text-xs text-slate-500"
              >
                Loading conversation…
              </div>
            ) : id && removed ? (
              <p className="py-10 text-sm text-slate-500">
                This conversation was deleted.
              </p>
            ) : Result.isFailure(watched) ? (
              <ErrorNotice
                title="Couldn’t open this chat"
                description={errorText(watched) ?? "Please try again."}
              />
            ) : !messages.length ? (
              <div className="flex h-full min-h-40 items-center justify-center">
                <MagnitudeMark className="size-12 opacity-20" />
              </div>
            ) : (
              <div
                ref={pinTranscript}
                className="mx-auto max-w-[760px] space-y-8 pb-8 pt-7 pr-1"
              >
                {messages.map((message, index) =>
                  message._tag === "user" ? (
                    <div key={message.id} className="flex justify-end">
                      <div className="max-w-[88%] whitespace-pre-wrap break-words rounded-2xl bg-slate-150 px-4 py-3 text-sm leading-6 dark:bg-slate-800">
                        {message.text}
                      </div>
                    </div>
                  ) : (
                    <ChatResponse
                      key={message.id}
                      response={message}
                      retry={
                        index === messages.length - 1 && !active
                          ? () => void submit(message.id)
                          : undefined
                      }
                      notify={setNotice}
                    />
                  )
                )}
              </div>
            )}
          </div>
          <div className="mx-auto w-full max-w-[760px] shrink-0 pb-4 pt-3">
            {active && chat && (
              <ChatRequestStatus
                response={active}
                live={liveResponse(chat, active)}
              />
            )}
            {(notice || error) && (
              <div
                role="alert"
                className="mb-3 flex items-start justify-between gap-3 text-xs leading-5 text-red-600 dark:text-red-400"
              >
                <span>{notice ?? error}</span>
                <button
                  type="button"
                  aria-label="Dismiss notice"
                  onClick={() => {
                    setNotice(null)
                    clearError()
                  }}
                >
                  <XIcon />
                </button>
              </div>
            )}
            {Result.isFailure(modelResult) && (
              <p role="alert" className="mb-3 text-xs text-red-600">
                {errorText(modelResult)}
              </p>
            )}
            {Result.isSuccess(modelResult) && !models.length && (
              <p className="mb-3 text-xs text-slate-500">
                Install a text model from Discover to start chatting.
              </p>
            )}
            {!settingsValid && (
              <p
                role="alert"
                className="mb-3 text-xs text-red-600 dark:text-red-400"
              >
                Check the values in Chat settings before sending.
              </p>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault()
                void submit()
              }}
              className="chat-composer rounded-xl border border-slate-300 bg-white p-3 transition-colors dark:border-slate-700 dark:bg-slate-850"
            >
              <textarea
                ref={input}
                // A chat's view mounts whenever a chat is opened or started, so its composer starts focused.
                autoFocus
                rows={2}
                maxLength={1_048_576}
                aria-label="Message"
                placeholder="Message…"
                value={composerText}
                onChange={(e) =>
                  setDraft({ ...draft, text: e.target.value, submitted: null })
                }
                onKeyDown={(e) => {
                  if (
                    submitsChatMessage(
                      e.key,
                      e.shiftKey,
                      e.nativeEvent.isComposing,
                      e.keyCode
                    )
                  ) {
                    e.preventDefault()
                    void submit()
                  }
                }}
                className="block max-h-48 min-h-14 w-full resize-none bg-transparent px-1 py-1 text-sm leading-6 outline-none dark:placeholder:text-slate-500"
              />
              <div className="mt-3 flex items-end justify-between gap-2">
                <div className="flex min-w-0 flex-wrap items-center gap-1">
                  <Select
                    value={modelId ?? ""}
                    onValueChange={(next) => {
                      if (next) {
                        const m = models.find((m) => m.id === next)
                        patch({
                          model: Option.some(ServingModelId.make(next)),
                          reasoningEffort: reasoningEffortFor(m, effort),
                        })
                      }
                    }}
                  >
                    <SelectTrigger
                      aria-label="Model"
                      disabled={!models.length}
                      variant="inline"
                      className="h-8 max-w-[230px] px-2 text-xs text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                    >
                      <SelectValue>
                        {model && (
                          <NamedLogo
                            name={model.name}
                            lab={labForName(model.name)}
                            className="size-3"
                          />
                        )}
                        <span className="truncate">
                          {model?.name ??
                            (modelId ? "Model unavailable" : "Select model")}
                        </span>
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent
                      side="top"
                      align="start"
                      alignItemWithTrigger={false}
                      className="w-72"
                    >
                      {models.map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          <div className="flex flex-col gap-1 py-0.5">
                            <span className="inline-flex items-center gap-1.5 text-[13px]">
                              <NamedLogo
                                name={m.name}
                                lab={labForName(m.name)}
                                className="size-3"
                              />
                              {m.name}
                            </span>
                            <span className="pl-[18px] text-xs font-normal text-slate-500">
                              {[
                                localFor(m.id)?.presentation.variantLabel,
                                `${number(contextFor(m))} context`,
                                localFor(m.id) &&
                                Option.exists(
                                  localModelResidency(localFor(m.id)!),
                                  (r) => r._tag === "Ready"
                                )
                                  ? "Loaded"
                                  : null,
                              ]
                                .filter(Boolean)
                                .join(" · ")}
                            </span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <ChatThinkingSelect
                    model={model}
                    selected={settings.reasoningEffort}
                    onChange={(next) =>
                      patch({ reasoningEffort: Option.some(next) })
                    }
                  />
                </div>
                {active ? (
                  <Button
                    type="button"
                    size="icon"
                    aria-label="Stop generation"
                    disabled={stopState.waiting}
                    onClick={() =>
                      id && stop({ chatId: id, responseId: active.id })
                    }
                  >
                    <SquareIcon weight="fill" />
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    size="icon"
                    aria-label="Send message"
                    disabled={!canSend}
                  >
                    {pending ? (
                      <CircleNotchIcon className="motion-safe:animate-spin" />
                    ) : (
                      <ArrowUpIcon weight="bold" />
                    )}
                  </Button>
                )}
              </div>
            </form>
          </div>
        </div>
        {panel && !narrow && (
          <aside
            className="w-[280px] shrink-0 overflow-y-auto border-l border-slate-200 py-5 pl-6 dark:border-slate-750"
            aria-label={panel === "settings" ? "Chat settings" : "Chat history"}
          >
            {body}
          </aside>
        )}
        {narrow && (
          <Dialog
            open={panel !== null}
            onOpenChange={(open) => {
              if (!open) setPanel(null)
            }}
          >
            <DialogContent
              className="chat-mobile-panel"
              showCloseButton={false}
            >
              {body}
            </DialogContent>
          </Dialog>
        )}
      </div>
    </section>
  )
}
const unwatched = Atom.make({
  result: Result.initial<WatchedChat>(),
  fetchStatus: "idle" as const,
  isStale: false,
  dataUpdatedAt: Option.none<number>(),
  failureCount: 0,
})
