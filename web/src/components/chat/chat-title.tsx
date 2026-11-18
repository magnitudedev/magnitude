import { useState } from "react"
import { useAtomSet } from "@effect-atom/atom-react"
import { Option } from "effect"
import { useAgentClient } from "@magnitudedev/client-common"
import type { SessionId } from "@magnitudedev/sdk"
import { CHAT_TITLE_MAX_LENGTH, renameOf } from "./rename"

const TITLE_CLASS = "min-w-0 font-heading text-[17px] font-semibold leading-7 tracking-tight"

/** The open chat's title above its transcript; double-clicking it renames the chat in place. */
export function ChatTitle({ chat }: { chat: { readonly id: SessionId; readonly title: string } | null }) {
  const client = useAgentClient()
  const rename = useAtomSet(client.Chat.RenameChat)
  const [editing, setEditing] = useState<{ readonly chatId: SessionId; readonly from: string; readonly title: string } | null>(null)
  const current = editing !== null && chat !== null && editing.chatId === chat.id ? editing : null
  const save = () => {
    if (current !== null) Option.map(renameOf({ chatId: current.chatId, from: current.from }, current.title), rename)
    setEditing(null)
  }
  if (current !== null) return <input
    autoFocus
    aria-label="Chat name"
    maxLength={CHAT_TITLE_MAX_LENGTH}
    value={current.title}
    onFocus={(event) => event.currentTarget.select()}
    onChange={(event) => setEditing({ ...current, title: event.target.value })}
    onBlur={save}
    onKeyDown={(event) => {
      if (event.key === "Enter") save()
      if (event.key === "Escape") setEditing(null)
    }}
    className={`${TITLE_CLASS} -mx-1.5 min-w-16 max-w-full rounded-md [field-sizing:content] border border-blue-500 bg-transparent px-1.5 outline-none`}
  />
  return <h1
    title={chat ? "Double-click to rename" : undefined}
    onDoubleClick={() => { if (chat) setEditing({ chatId: chat.id, from: chat.title, title: chat.title }) }}
    className={`${TITLE_CLASS} truncate border border-transparent ${chat ? "cursor-default select-none" : ""}`}
  >
    {chat?.title ?? "Chat"}
  </h1>
}
