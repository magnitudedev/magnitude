import { submitsChatMessage } from "./chat/keyboard"
import { useContext, useRef, useState } from "react"
import {
  Atom,
  RegistryContext,
  Result,
  useAtomSet,
  useAtomValue,
} from "@effect-atom/atom-react"
import { Option, Schema } from "effect"
import {
  ChatCommandId,
  ChatConversationId,
  ChatModelId,
  ChatSettings as ChatSettingsSchema,
  defaultChatSettings,
  type ChatConversation,
  type ChatGeneration,
  type ChatSettings,
} from "@magnitudedev/sdk"
import {
  chatDraft,
  chatComposerText,
  completeChatSubmission,
  selectedChat,
  chatReasoningLevels,
  chatReasoningEffort,
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
  ChartBarIcon,
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
  titleCase,
  value,
} from "./chat/shared"
import { ChatResponse } from "./chat/response"
import { ChatRequestStatus } from "./chat/activity"
import { ChatSettingsPanel } from "./chat/settings"
import { ChatHistoryPanel } from "./chat/history"

export function ChatPage() {
  const id = useAtomValue(selectedChat)
  const select = useAtomSet(selectedChat)
  return <ChatView key={id ?? "new"} id={id} select={select} />
}
function ChatView({
  id,
  select,
}: {
  id: ChatConversationId | null
  select: (id: ChatConversationId | null) => void
}) {
  const client = useAgentClient()
  const registry = useContext(RegistryContext)
  const clearCreate = useAtomSet(client.Chat.CreateConversation)
  const clearSend = useAtomSet(client.Chat.StartGeneration)
  const clearStop = useAtomSet(client.Chat.StopGeneration)
  const clearError = () => {
    clearCreate(Atom.Reset)
    clearSend(Atom.Reset)
    clearStop(Atom.Reset)
  }
  const [panel, setPanel] = useState<"settings" | "history" | null>(null)
  const draftAtom = chatDraft(id)
  const draft = useAtomValue(draftAtom)
  const setDraft = useAtomSet(draftAtom)
  const setNewDraft = useAtomSet(chatDraft(null))
  // A missing conversation uses an inert atom; no request is made for a new draft.
  const watched = useAtomValue(
    id
      ? client.Chat.WatchConversation({ conversationId: id })
      : emptyConversation
  ).result
  const conversation = Result.isSuccess(watched) ? watched.value : null
  const modelResult = useAtomValue(client.Chat.ListModels({})).result
  const models = Result.isSuccess(modelResult)
    ? modelResult.value.data.filter(
        (m) =>
          m.architecture.input_modalities.includes("text") &&
          m.architecture.output_modalities.includes("text")
      )
    : []
  const settings = {
    ...(conversation?.settings ?? defaultChatSettings),
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
  const modelId = value(settings.modelId) ?? models[0]?.id
  const model = models.find((m) => m.id === modelId)
  const reasoning = model ? Option.getOrNull(model.reasoning) : null
  const levels = chatReasoningLevels(model)
  const effort = value(chatReasoningEffort(model, settings.reasoningEffort))
  const effectiveSettings: ChatSettings = {
    ...settings,
    modelId: modelId ? Option.some(ChatModelId.make(modelId)) : none(),
    reasoningEffort: chatReasoningEffort(model, settings.reasoningEffort),
  }
  const settingsValid = Schema.is(Schema.typeSchema(ChatSettingsSchema))(
    effectiveSettings
  )
  const patch = (change: Partial<ChatSettings>) =>
    setDraft({ ...draft, settings: { ...draft.settings, ...change } })
  const create = useAtomSet(client.Chat.CreateConversation, { mode: "promise" })
  const send = useAtomSet(client.Chat.StartGeneration, { mode: "promise" })
  const stop = useAtomSet(client.Chat.StopGeneration)
  const createState = useAtomValue(client.Chat.CreateConversation)
  const sendState = useAtomValue(client.Chat.StartGeneration)
  const stopState = useAtomValue(client.Chat.StopGeneration)
  const pending = createState.waiting || sendState.waiting
  const active = conversation?.messages
    .flatMap((m) => Option.toArray(m.generation))
    .find((g) => g.status === "running")
  const input = useRef<HTMLTextAreaElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const follow = useRef(true)
  const [notice, setNotice] = useState<string | null>(null)
  const composerText = chatComposerText(
    draft,
    conversation?.messages.at(-1)?.generation.pipe(Option.getOrUndefined)
  )
  const submit = async (retryMessageId?: ChatGeneration["messageId"]) => {
    if (
      active ||
      pending ||
      !settingsValid ||
      !modelId ||
      (!retryMessageId && !composerText.trim())
    )
      return
    const text = retryMessageId
      ? conversation?.messages.at(-2)?.content ?? ""
      : composerText
    follow.current = true
    try {
      const conversationId = id ?? ChatConversationId.make(crypto.randomUUID())
      if (!id) await create({ conversationId, settings: effectiveSettings })
      const generationId = await send({
        conversationId,
        commandId: ChatCommandId.make(crypto.randomUUID()),
        content: retryMessageId ? none() : Option.some(text),
        retryMessageId: Option.fromNullable(retryMessageId),
        settings: effectiveSettings,
      })
      const latest = registry.get(draftAtom)
      const next = completeChatSubmission(
        latest,
        text,
        generationId,
        !!retryMessageId
      )
      setDraft(next)
      if (!id) {
        registry.set(chatDraft(conversationId), next)
        setNewDraft({ ...latest, text: "", submitted: null })
        if (registry.get(selectedChat) === id) select(conversationId)
      }
      input.current?.focus()
    } catch {
      /* The mutation's typed failure is rendered below; the draft stays intact. */
    }
  }
  const error = pending
    ? null
    : errorText(createState) ?? errorText(sendState) ?? errorText(stopState)
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
              ...defaultChatSettings,
              modelId: effectiveSettings.modelId,
              reasoningEffort: reasoning
                ? Option.some(reasoning.default_effort)
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
            onScroll={() => {
              const node = scroller.current
              if (node)
                follow.current =
                  node.scrollHeight - node.clientHeight - node.scrollTop < 80
            }}
            className="chat-scroll min-h-0 flex-1 overflow-y-auto"
          >
            {id && Result.isInitial(watched) ? (
              <div
                role="status"
                className="py-10 text-center text-xs text-slate-500"
              >
                Loading conversation…
              </div>
            ) : Result.isFailure(watched) ? (
              <ErrorNotice
                title="Couldn’t open this chat"
                description={errorText(watched) ?? "Please try again."}
              />
            ) : id && Result.isSuccess(watched) && !conversation ? (
              <p className="py-10 text-sm text-slate-500">
                This conversation was deleted.
              </p>
            ) : !conversation?.messages.length ? (
              <div className="flex h-full min-h-40 items-center justify-center">
                <MagnitudeMark className="size-12 opacity-20" />
              </div>
            ) : (
              <div
                ref={(node) => {
                  if (node && follow.current && scroller.current)
                    scroller.current.scrollTop = scroller.current.scrollHeight
                }}
                className="mx-auto max-w-[760px] space-y-8 pb-8 pt-7 pr-1"
              >
                {conversation.messages.map((message, index) =>
                  message.role === "user" ? (
                    <div key={message.id} className="flex justify-end">
                      <div className="max-w-[88%] whitespace-pre-wrap break-words rounded-2xl bg-slate-150 px-4 py-3 text-sm leading-6 dark:bg-slate-800">
                        {message.content}
                      </div>
                    </div>
                  ) : Option.isSome(message.generation) ? (
                    <ChatResponse
                      key={message.generation.value.id}
                      generation={message.generation.value}
                      retry={
                        index === conversation.messages.length - 1 && !active
                          ? () => void submit(message.id)
                          : undefined
                      }
                      notify={setNotice}
                    />
                  ) : null
                )}
              </div>
            )}
          </div>
          <div className="mx-auto w-full max-w-[760px] shrink-0 pb-4 pt-3">
            {active && <ChatRequestStatus generation={active} />}
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
                Install a text model from Discover or Catalog to start chatting.
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
                        const r = m ? Option.getOrNull(m.reasoning) : null
                        patch({
                          modelId: Option.some(ChatModelId.make(next)),
                          reasoningEffort: chatReasoningEffort(
                            m,
                            Option.fromNullable(effort)
                          ),
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
                  {reasoning && levels.length > 1 && (
                    <Select
                      value={effort}
                      onValueChange={(next) => {
                        if (next) patch({ reasoningEffort: Option.some(next) })
                      }}
                    >
                      <SelectTrigger
                        aria-label="Thinking level"
                        variant="inline"
                        className="h-8 px-2 text-xs text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
                      >
                        <SelectValue>
                          <ChartBarIcon aria-hidden="true" className="size-3" />
                          {titleCase(effort ?? reasoning.default_effort)}
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent
                        side="top"
                        align="start"
                        alignItemWithTrigger={false}
                        className="w-36"
                      >
                        {levels.map((level) => (
                          <SelectItem key={level.value} value={level.value}>
                            {level.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </div>
                {active ? (
                  <Button
                    type="button"
                    size="icon"
                    aria-label="Stop generation"
                    disabled={stopState.waiting}
                    onClick={() =>
                      id &&
                      stop({ conversationId: id, generationId: active.id })
                    }
                  >
                    <SquareIcon weight="fill" />
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    size="icon"
                    aria-label="Send message"
                    disabled={
                      !composerText.trim() ||
                      !model ||
                      pending ||
                      !settingsValid ||
                      (!!id && !conversation)
                    }
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
const emptyConversation = Atom.make({
  result: Result.success<ChatConversation | null>(null),
  fetchStatus: "idle" as const,
  isStale: false,
  dataUpdatedAt: Option.none<number>(),
  failureCount: 0,
})
