import { useEffect, useRef, useState, type FormEvent } from "react"
import { ArrowUpIcon, PlusIcon, StopIcon, TrashIcon } from "@phosphor-icons/react"
import { Button } from "./ui/button"

type ChatMessage = { role: "user" | "assistant"; content: string }
type Conversation = {
  id: string
  title: string
  model: string
  messages: ChatMessage[]
  updatedAt: number
}

const STORAGE_KEY = "magnitude.web.chat.v1"
const apiBase = "/inference/v1"

function readConversations(): Conversation[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]")
    if (!Array.isArray(value)) return []
    return value.filter((item): item is Conversation =>
      typeof item?.id === "string" &&
      typeof item?.title === "string" &&
      typeof item?.model === "string" &&
      Array.isArray(item?.messages) &&
      item.messages.every((message: unknown) =>
        typeof message === "object" && message !== null &&
        ((message as ChatMessage).role === "user" || (message as ChatMessage).role === "assistant") &&
        typeof (message as ChatMessage).content === "string"
      )
    )
  } catch {
    return []
  }
}

function saveConversations(conversations: Conversation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
  } catch {
    // Storage can be disabled or full. The active conversation remains usable in memory.
  }
}

function consumeEvent(block: string, onText: (text: string) => void): boolean {
  let finished = false
  for (const line of block.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue
    const data = line.slice(5).trimStart()
    if (data === "[DONE]") {
      finished = true
      continue
    }
    if (!data) continue
    try {
      const event = JSON.parse(data) as {
        choices?: Array<{ delta?: { content?: unknown } }>
        error?: { message?: string }
      }
      if (event.error?.message) throw new Error(event.error.message)
      const content = event.choices?.[0]?.delta?.content
      if (typeof content === "string" && content.length > 0) onText(content)
    } catch (error) {
      if (error instanceof Error && error.message !== "Unexpected end of JSON input") throw error
      // Ignore non-JSON keepalive/comment events; malformed JSON data frames are reported by the server.
    }
  }
  return finished
}

export function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>(readConversations)
  const [activeId, setActiveId] = useState<string | null>(() => readConversations()[0]?.id ?? null)
  const [models, setModels] = useState<string[]>([])
  const [preferredModel, setPreferredModel] = useState("")
  const [modelsLoading, setModelsLoading] = useState(true)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [draft, setDraft] = useState("")
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [storageWarning, setStorageWarning] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const active = conversations.find(item => item.id === activeId) ?? null
  const selectedModel = (active?.model && models.includes(active.model) ? active.model : "") || preferredModel || models[0] || ""

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      setModelsLoading(true)
      setModelsError(null)
      try {
        const response = await fetch(`${apiBase}/models`)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const body = await response.json() as { data?: Array<{ id?: unknown }> }
        const ids = (body.data ?? []).flatMap(item => typeof item.id === "string" ? [item.id] : [])
        if (!cancelled) {
          setModels(ids)
          if (ids.length === 0) setModelsError("Magnitude has no models available for inference yet. Install or load a model, then retry.")
        }
      } catch (cause) {
        if (!cancelled) setModelsError(`Could not load local models: ${cause instanceof Error ? cause.message : String(cause)}`)
      } finally {
        if (!cancelled) setModelsLoading(false)
      }
    }
    void load()
    return () => { cancelled = true; abortRef.current?.abort() }
  }, [])

  useEffect(() => {
    saveConversations(conversations)
    try {
      setStorageWarning(false)
      // Detect storage denial explicitly; saveConversations deliberately fails closed.
      localStorage.setItem(`${STORAGE_KEY}.probe`, "1")
      localStorage.removeItem(`${STORAGE_KEY}.probe`)
    } catch {
      setStorageWarning(true)
    }
  }, [conversations])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }) }, [active?.messages.length, active?.messages.at(-1)?.content])

  const createConversation = () => {
    const conversation: Conversation = {
      id: crypto.randomUUID(),
      title: "New chat",
      model: selectedModel,
      messages: [],
      updatedAt: Date.now(),
    }
    const next = [conversation, ...conversations]
    setConversations(next)
    setActiveId(conversation.id)
    setError(null)
  }

  const updateConversation = (conversation: Conversation) => {
    const next = [conversation, ...conversations.filter(item => item.id !== conversation.id)]
    next.sort((a, b) => b.updatedAt - a.updatedAt)
    setConversations(next)
  }

  const send = async (event: FormEvent) => {
    event.preventDefault()
    const text = draft.trim()
    if (!text || !selectedModel || working) return
    let conversation = active
    if (!conversation) {
      conversation = {
        id: crypto.randomUUID(), title: "New chat", model: selectedModel,
        messages: [], updatedAt: Date.now(),
      }
      setActiveId(conversation.id)
    }
    conversation.model = selectedModel
    if (conversation.title === "New chat") conversation.title = text.replace(/\s+/g, " ").slice(0, 54)
    conversation.messages = [...conversation.messages, { role: "user", content: text }, { role: "assistant", content: "" }]
    conversation.updatedAt = Date.now()
    updateConversation(conversation)
    setDraft("")
    setError(null)
    setWorking(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const response = await fetch(`${apiBase}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: selectedModel, messages: conversation.messages.slice(0, -1), stream: true }),
        signal: controller.signal,
      })
      if (!response.ok) {
        let detail = `HTTP ${response.status}`
        try {
          const body = await response.json() as { error?: { message?: string } }
          if (body.error?.message) detail = body.error.message
        } catch { /* Keep the HTTP status as the useful fallback. */ }
        throw new Error(detail)
      }
      if (!response.body) throw new Error("The inference response did not provide a stream.")
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let done = false
      const append = (delta: string) => {
        const last = conversation!.messages[conversation!.messages.length - 1]
        last.content += delta
        conversation!.updatedAt = Date.now()
        updateConversation(conversation!)
      }
      while (!done) {
        const part = await reader.read()
        done = part.done
        buffer += decoder.decode(part.value, { stream: !done })
        buffer = buffer.replace(/\r\n/g, "\n")
        const blocks = buffer.split("\n\n")
        buffer = blocks.pop() ?? ""
        for (const block of blocks) {
          consumeEvent(block, append)
          if (controller.signal.aborted) break
        }
      }
      if (buffer.trim()) consumeEvent(buffer, append)
      await reader.cancel().catch(() => undefined)
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        const message = cause instanceof Error ? cause.message : String(cause)
        setError(`Could not complete the request: ${message}`)
      }
    } finally {
      abortRef.current = null
      setWorking(false)
      conversation.updatedAt = Date.now()
      updateConversation(conversation)
    }
  }

  const removeConversation = (id: string) => {
    const next = conversations.filter(item => item.id !== id)
    setConversations(next)
    if (activeId === id) setActiveId(next[0]?.id ?? null)
  }

  return <div className="flex min-h-[calc(100vh-8rem)] min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-750 dark:bg-slate-925">
    <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-200 bg-slate-50 p-3 dark:border-slate-750 dark:bg-slate-900 sm:flex">
      <Button onClick={createConversation} className="mb-3 justify-start gap-2"><PlusIcon className="size-4" />New chat</Button>
      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto" aria-label="Conversation history">
        {conversations.map(item => <div key={item.id} className={`group flex items-center gap-1 rounded-lg ${item.id === activeId ? "bg-blue-50 dark:bg-slate-800" : ""}`}>
          <button type="button" onClick={() => { setActiveId(item.id); setError(null) }} className="min-w-0 flex-1 truncate rounded-lg px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-blue-500 dark:text-slate-200 dark:hover:bg-slate-800" aria-current={item.id === activeId ? "page" : undefined}>{item.title}</button>
          <button type="button" onClick={() => removeConversation(item.id)} aria-label={`Delete ${item.title}`} className="mr-1 rounded p-1.5 text-slate-400 opacity-0 hover:text-red-600 focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-blue-500 group-hover:opacity-100"><TrashIcon className="size-4" /></button>
        </div>)}
        {conversations.length === 0 && <p className="px-3 py-2 text-xs text-slate-500">Your conversations will appear here.</p>}
      </div>
      <p className="px-2 pt-3 text-xs text-slate-500">Stored in this browser only.</p>
    </aside>
    <section className="flex min-w-0 flex-1 flex-col">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-3 dark:border-slate-750 sm:px-6">
        <div className="min-w-0"><h2 className="truncate text-sm font-semibold">{active?.title ?? "New chat"}</h2><p className="mt-0.5 text-xs text-slate-500">Local inference · streaming responses</p></div>
        <label className="flex items-center gap-2 text-xs text-slate-500">Model
          <select aria-label="Model" value={selectedModel} disabled={modelsLoading || models.length === 0 || working} onChange={event => {
            const model = event.target.value
            if (!active) { setPreferredModel(model); return }
            updateConversation({ ...active, model, updatedAt: Date.now() })
          }} className="max-w-[min(52vw,22rem)] rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus-visible:outline-2 focus-visible:outline-blue-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100">
            {models.length === 0 && <option value="">No models available</option>}
            {models.map(model => <option key={model} value={model}>{model}</option>)}
          </select>
        </label>
      </header>
      {modelsError && <div role="alert" className="m-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">{modelsError} <button type="button" className="ml-2 underline" onClick={() => window.location.reload()}>Retry</button></div>}
      {storageWarning && <p role="status" className="px-5 pt-3 text-xs text-amber-700 dark:text-amber-300">Browser storage is unavailable; history may be lost when you leave this page.</p>}
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-5 sm:px-8" aria-label="Conversation messages" aria-live="polite">
        {!active?.messages.length && <div className="mx-auto flex min-h-64 max-w-xl flex-col items-center justify-center text-center">
          <div className="mb-3 text-3xl text-blue-600">◈</div><h3 className="text-2xl font-semibold tracking-tight">Chat with your local model</h3>
          <p className="mt-2 text-sm leading-6 text-slate-500">Choose a model and ask a question. Text appears as it is generated. Your conversation history stays in this browser.</p>
          {!modelsLoading && models.length === 0 && <p className="mt-3 text-sm text-slate-500">Load a model in My Models before starting a conversation.</p>}
        </div>}
        {active?.messages.map((message, index) => <article key={`${active.id}-${index}`} className={`mx-auto w-full max-w-3xl ${message.role === "user" ? "rounded-xl border border-blue-100 bg-blue-50 px-4 py-3 dark:border-blue-950 dark:bg-blue-950/40" : "px-1 py-2"}`}>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">{message.role === "user" ? "You" : "Magnitude"}</p>
          <p className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-800 dark:text-slate-100">{message.content || (working && index === active.messages.length - 1 ? "Generating…" : "")}</p>
        </article>)}
        <div ref={endRef} />
      </div>
      {error && <div role="alert" className="mx-4 mb-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">{error}</div>}
      <form onSubmit={send} className="border-t border-slate-200 p-3 dark:border-slate-750 sm:p-5">
        <div className="mx-auto max-w-3xl rounded-xl border border-slate-300 bg-white p-2 focus-within:border-blue-400 dark:border-slate-700 dark:bg-slate-900">
          <textarea value={draft} onChange={event => setDraft(event.target.value)} onKeyDown={event => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); event.currentTarget.form?.requestSubmit() }
          }} rows={2} aria-label="Message" placeholder="Message your local model…" className="max-h-48 min-h-16 w-full resize-y bg-transparent px-2 py-2 text-sm outline-none placeholder:text-slate-400" disabled={working || models.length === 0} />
          <div className="flex items-center justify-between gap-3 px-1 pb-1">
            <span className="text-xs text-slate-500">Enter to send · Shift+Enter for a new line</span>
            {working
              ? <Button type="button" variant="outline" onClick={() => abortRef.current?.abort()} className="gap-2"><StopIcon className="size-4" />Stop</Button>
              : <Button type="submit" disabled={!draft.trim() || !selectedModel || modelsLoading} className="gap-2"><ArrowUpIcon className="size-4" />Send</Button>}
          </div>
        </div>
      </form>
    </section>
  </div>
}
