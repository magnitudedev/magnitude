import { submitsChatMessage } from "./chat/keyboard"
import { useCallback, useContext, useRef, useState } from "react"
import {
  Atom,
  RegistryContext,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Cause, DateTime, Effect, Option, Schema } from "effect"
import {
  chatMessages,
  generateSessionId,
  type InferenceModel,
  liveResponse,
  runningResponse,
  type MessageId,
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
  chatStartingSettings,
  completeChatSubmission,
  localModelResidency,
  reasoningEffortFor,
  selectedChat,
  type ChatRequestSettings,
  useAgentClient,
} from "@magnitudedev/client-common"
import {
  ArrowUpIcon,
  PlusIcon,
  SquareIcon,
  CircleNotchIcon,
} from "@phosphor-icons/react"
import { Button } from "./ui/button"
import { MagnitudeMark } from "./magnitude-mark"
import { ErrorNotice } from "./error-notice"
import { CopyText, SentAt, errorText, none, number } from "./chat/shared"
import { ComposerAlert, composerError } from "./chat/composer-alert"
import { ChatResponse } from "./chat/response"
import { ChatRequestStatus } from "./chat/activity"
import { ChatTitle } from "./chat/chat-title"
import { ModelPicker, type PickerModel } from "./chat/model-picker"
import { useChatModels } from "./chat/served-models"
import { submitBlocked } from "./chat/submit-blocked"
import { useNavigate } from "../session"

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
  const navigate = useNavigate()
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
  const { result: modelResult, models, localFor, nameOf, contextFor, contextLimitFor } = useChatModels()
  const savedResult = useAtomValue(client.Chat.ListChatModelDefaults({})).result
  const saved = Result.isSuccess(savedResult) ? savedResult.value : {}
  const transcript = chatRequestSettings(chat?.messages ?? [])
  const modelId =
    Option.getOrUndefined(draft.settings.model ?? transcript.model) ?? models[0]?.id
  const model = models.find((m) => m.id === modelId)
  // Each model's settings live in Settings → Chat; a chat chooses its model and thinking level, keeping the level
  // its latest response used until it chooses another.
  const continued = Option.isSome(transcript.model) && transcript.model.value === modelId
  const settings: ChatRequestSettings = {
    ...(model ? chatStartingSettings(model, saved) : transcript),
    ...(continued ? { reasoningEffort: transcript.reasoningEffort } : {}),
    ...draft.settings,
  }
  const effectiveSettings: ChatRequestSettings = {
    ...settings,
    model: modelId ? Option.some(ServingModelId.make(modelId)) : none(),
    reasoningEffort: reasoningEffortFor(model, settings.reasoningEffort),
  }
  const request = chatRequest(effectiveSettings)
  const settingsValid = Schema.is(Schema.typeSchema(RequestedResponseSettings))(
    effectiveSettings
  )
  // Choosing commits the chat to the model it shows, so a model list that changes meanwhile can't move it.
  const patch = (change: Partial<ChatRequestSettings>) =>
    setDraft({
      ...draft,
      settings: { ...draft.settings, model: effectiveSettings.model, ...change },
    })
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
  const active = Option.getOrUndefined(runningResponse(messages))
  const input = useRef<HTMLTextAreaElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  /** A new chat with this model, its draft `text`: empty, or the message a chat that outgrew its context couldn't answer. */
  const startNewChat = (text = "") => {
    clearError()
    select(null)
    // The new chat keeps this model and starts from its settings.
    setNewDraft({
      text,
      settings: { model: effectiveSettings.model },
      submitted: null,
    })
    input.current?.focus()
  }
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
    if (Option.isNone(ready)) {
      // Nothing is sent; say why when it is the model or its settings, which the reader can change.
      Option.match(submitBlocked({ chosen: modelId !== undefined, available: model !== undefined, settingsValid }), { onNone: () => undefined, onSome: setNotice })
      return
    }
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
  // Settings and models often fail for the same reason, such as the service being down; one line says it once, and
  // not again when the chat that couldn't open already says it.
  const error = composerError(pending, [startState, sendState, retryState, stopState, savedResult, modelResult], errorText(watched))
  const shownAlert = notice ?? error
  const pickerNote = (m: InferenceModel) => {
    const found = localFor(m.id)
    const limit = contextLimitFor(m)
    return [
      limit === undefined ? null : `${number(contextFor(m))} of ${number(limit)} context`,
      found && Option.exists(localModelResidency(found), (r) => r._tag === "Ready") ? "Loaded" : null,
      m.id in saved ? "Your defaults" : null,
    ].filter(Boolean).join(" · ") || null
  }
  const pickerModels: readonly PickerModel[] = models.map((m) => ({
    model: m,
    name: nameOf(m),
    context: contextFor(m),
    note: pickerNote(m),
  }))
  return (
    <section aria-label="Chat" className="flex min-h-0 flex-1 flex-col">
      <div className="mb-5 flex min-h-9 shrink-0 items-center justify-between gap-3">
        <ChatTitle chat={chat?.chat ?? null} />
        <Button variant="ghost" className="shrink-0" disabled={pending} onClick={() => startNewChat()}>
          <PlusIcon />
          New chat
        </Button>
      </div>
      <div className="relative flex min-h-0 flex-1">
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
                    <div key={message.id} className="chat-hover-group ml-auto flex w-fit max-w-[88%] flex-col items-end gap-2">
                      <div className="max-w-full whitespace-pre-wrap break-words rounded-2xl bg-slate-150 px-4 py-3 text-sm leading-6 dark:bg-slate-800">
                        {message.text}
                      </div>
                      <span className="chat-message-actions -mt-1">
                        {message.text.length > 0 && <CopyText text={message.text} label="Copy message" notify={setNotice} />}
                        <SentAt time={DateTime.toEpochMillis(message.time)} />
                      </span>
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
                      newChat={pending ? undefined : () => { const asked = messages[index - 1]; startNewChat(asked?._tag === "user" ? asked.text : "") }}
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
            {shownAlert && <ComposerAlert text={shownAlert} onDismiss={() => { setNotice(null); clearError() }} />}
            {Result.isSuccess(modelResult) && !models.length && (
              <p className="mb-3 text-xs text-slate-500">
                Download a model in{" "}
                <button type="button" onClick={() => navigate("models")} className="font-medium text-slate-700 hover:underline dark:text-slate-300">
                  Models
                </button>{" "}
                to start chatting.
              </p>
            )}
            {!settingsValid && (
              <p
                role="alert"
                className="mb-3 text-xs text-red-600 dark:text-red-400"
              >
                Check this model’s values in Settings → Chat before sending.
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
              <div className="mt-3 flex items-end justify-end gap-2">
                <ModelPicker
                  models={pickerModels}
                  model={model}
                  name={model ? nameOf(model) : modelId ? "Model unavailable" : "Select model"}
                  selected={settings.reasoningEffort}
                  onModel={(m) => patch({ model: Option.some(ServingModelId.make(m.id)), reasoningEffort: chatStartingSettings(m, saved).reasoningEffort })}
                  onEffort={(next) => patch({ reasoningEffort: Option.some(next) })}
                  onManage={() => navigate("models")}
                />
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
