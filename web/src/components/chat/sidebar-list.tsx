import { useEffect, useRef, useState } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Option } from "effect"
import type { ApplicationPage } from "@magnitudedev/client-common"
import { selectedChat, useAgentClient } from "@magnitudedev/client-common"
import type { ChatSummary } from "@magnitudedev/sdk"
import { DotsThreeIcon, PencilSimpleIcon, PlusIcon, TrashIcon } from "@phosphor-icons/react"
import { ConfirmDialog } from "../confirm-dialog"
import { CHAT_TITLE_MAX_LENGTH, renameOf } from "./rename"

const PAGE = 30

function Row({ chat, current, open }: { chat: ChatSummary; current: boolean; open: () => void }) {
  const client = useAgentClient()
  const rename = useAtomSet(client.Chat.RenameChat)
  const remove = useAtomSet(client.Chat.RemoveChat)
  const select = useAtomSet(selectedChat)
  const selected = useAtomValue(selectedChat)
  const [menu, setMenu] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [confirming, setConfirming] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!menu) return
    const close = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenu(false)
    }
    document.addEventListener("mousedown", close)
    return () => document.removeEventListener("mousedown", close)
  }, [menu])
  const save = (title: string) => {
    Option.map(renameOf({ chatId: chat.id, from: chat.title }, title), rename)
    setEditing(null)
  }
  return (
    <div className="chat-list-row" data-current={current} data-menu={menu}>
      {editing !== null ? (
        <input
          autoFocus
          aria-label="Chat name"
          maxLength={CHAT_TITLE_MAX_LENGTH}
          className="chat-list-rename"
          value={editing}
          onChange={(event) => setEditing(event.target.value)}
          onBlur={() => save(editing)}
          onKeyDown={(event) => {
            if (event.key === "Enter") save(editing)
            if (event.key === "Escape") setEditing(null)
          }}
        />
      ) : (
        <button type="button" className="chat-list-title" title={chat.title} onClick={open} aria-current={current ? "page" : undefined}>
          <span className="truncate">{chat.title}</span>
        </button>
      )}
      {editing === null && (
        <div className="relative" ref={menuRef}>
          <button type="button" aria-label={`Options for ${chat.title}`} aria-expanded={menu} className="chat-list-more" onClick={() => setMenu(!menu)}>
            <DotsThreeIcon weight="bold" />
          </button>
          {menu && (
            <div role="menu" className="chat-list-menu">
              <button type="button" role="menuitem" onClick={() => { setMenu(false); setEditing(chat.title) }}>
                <PencilSimpleIcon /> Rename
              </button>
              <button type="button" role="menuitem" className="chat-list-danger" onClick={() => { setMenu(false); setConfirming(true) }}>
                <TrashIcon /> Delete
              </button>
            </div>
          )}
        </div>
      )}
      <ConfirmDialog
        open={confirming}
        onOpenChange={setConfirming}
        title="Delete this chat?"
        description={`“${chat.title}” and its messages will be deleted.`}
        confirmLabel="Delete"
        destructive
        onConfirm={() => {
          remove({ chatId: chat.id })
          if (selected === chat.id) select(null)
          setConfirming(false)
        }}
      />
    </div>
  )
}

/** Every chat, newest first, a page at a time, loading the next page as the list's end comes into view; a quiet line until there is one. */
export function ChatList({ navigate, onNavigate }: { navigate: (page: ApplicationPage) => void; onNavigate?: () => void }) {
  const [pages, setPages] = useState(1)
  return (
    <div className="chat-list">
      {Array.from({ length: pages }, (_, index) => (
        <ChatListPage key={index} offset={index * PAGE} last={index === pages - 1} more={() => setPages(pages + 1)} navigate={navigate} onNavigate={onNavigate} />
      ))}
    </div>
  )
}

function ChatListPage({ offset, last, more, navigate, onNavigate }: {
  offset: number
  last: boolean
  more: () => void
  navigate: (page: ApplicationPage) => void
  onNavigate: (() => void) | undefined
}) {
  const client = useAgentClient()
  const listed = useAtomValue(client.Chat.ListChats({ offset, limit: PAGE })).result
  const selected = useAtomValue(selectedChat)
  const select = useAtomSet(selectedChat)
  const end = useRef<HTMLDivElement>(null)
  const hasMore = Result.isSuccess(listed) && listed.value.hasMore
  useEffect(() => {
    const node = end.current
    if (!last || !hasMore || node === null) return
    const observer = new IntersectionObserver((entries) => { if (entries.some((entry) => entry.isIntersecting)) more() })
    observer.observe(node)
    return () => observer.disconnect()
  }, [last, hasMore])
  if (!Result.isSuccess(listed)) return null
  if (offset === 0 && listed.value.chats.length === 0) return <p className="chat-list-empty">No chats yet. Click New chat above to start one.</p>
  return <>
    {listed.value.chats.map((chat) => (
      <Row
        key={chat.id}
        chat={chat}
        current={selected === chat.id}
        open={() => {
          select(chat.id)
          navigate("chat")
          onNavigate?.()
        }}
      />
    ))}
    {last && hasMore && <div ref={end} aria-hidden="true" className="h-px" />}
  </>
}

/** The top of the sidebar: New chat, and every chat beneath it once the app has reached the service. */
export function ChatSection({ navigate, onNavigate }: { navigate?: (page: ApplicationPage) => void; onNavigate?: () => void }) {
  const select = useAtomSet(selectedChat)
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <button type="button" className="new-chat-button group" disabled={!navigate} onClick={() => { select(null); navigate?.("chat"); onNavigate?.() }}>
        New chat<PlusIcon weight="bold" className="size-3.5 transition-transform duration-200 group-hover:rotate-90" />
      </button>
      {navigate && <ChatList navigate={navigate} onNavigate={onNavigate} />}
    </div>
  )
}
