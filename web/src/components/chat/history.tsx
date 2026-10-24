import { useState } from "react"
import { Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Option } from "effect"
import type { ChatConversationId, ChatSummary } from "@magnitudedev/sdk"
import { useAgentClient } from "@magnitudedev/client-common"
import { PencilSimpleIcon, TrashIcon } from "@phosphor-icons/react"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { IconButton, errorText, none } from "./shared"
export function ChatHistoryPanel({
  selected,
  open,
}: {
  selected: ChatConversationId | null
  open: (id: ChatConversationId | null) => void
}) {
  const client = useAgentClient()
  const [offset, setOffset] = useState(0)
  const limit = 25
  const history = useAtomValue(
    client.Chat.ListConversations({ offset, limit })
  ).result
  const update = useAtomSet(client.Chat.UpdateConversation)
  const remove = useAtomSet(client.Chat.DeleteConversation, { mode: "promise" })
  const updateState = useAtomValue(client.Chat.UpdateConversation)
  const deleteState = useAtomValue(client.Chat.DeleteConversation)
  const [editing, setEditing] = useState<ChatConversationId | null>(null)
  const [rename, setRename] = useState("")
  const [deleting, setDeleting] = useState<ChatConversationId | null>(null)
  const groups = new Map<string, ChatSummary[]>()
  if (Result.isSuccess(history))
    for (const chat of history.value.conversations) {
      const date = new Date(chat.updatedAt)
      const today = new Date()
      const yesterday = new Date(today)
      yesterday.setDate(today.getDate() - 1)
      const label =
        date.toDateString() === today.toDateString()
          ? "Today"
          : date.toDateString() === yesterday.toDateString()
          ? "Yesterday"
          : date.toLocaleDateString()
      groups.set(label, [...(groups.get(label) ?? []), chat])
    }
  return (
    <>
      <p className="mb-5 mt-1 text-xs text-slate-500">
        Saved on the inference service
      </p>
      {Result.isInitial(history) && (
        <p className="text-xs text-slate-500">Loading chats…</p>
      )}
      {Result.isSuccess(history) &&
        history.value.conversations.length === 0 && (
          <p className="text-sm text-slate-500">
            {offset === 0
              ? "No saved chats yet."
              : "No older chats on this page."}
          </p>
        )}
      {(errorText(history) ??
        errorText(updateState) ??
        errorText(deleteState)) && (
        <p role="alert" className="mb-4 text-xs text-red-600">
          {errorText(history) ??
            errorText(updateState) ??
            errorText(deleteState)}
        </p>
      )}
      {[...groups].map(([day, chats]) => (
        <div key={day} className="mb-6">
          <div className="mb-2 text-xs text-slate-500">{day}</div>
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={`group rounded-lg px-3 py-3 hover:bg-slate-150 dark:hover:bg-slate-800 ${
                selected === chat.id ? "bg-slate-100 dark:bg-slate-850" : ""
              }`}
            >
              {editing === chat.id ? (
                <form
                  onSubmit={(e) => {
                    e.preventDefault()
                    if (rename.trim()) {
                      update({
                        conversationId: chat.id,
                        title: Option.some(rename.trim()),
                        settings: none(),
                      })
                      setEditing(null)
                    }
                  }}
                >
                  <Input
                    autoFocus
                    aria-label="Chat title"
                    maxLength={200}
                    value={rename}
                    onChange={(e) => setRename(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Escape") setEditing(null)
                    }}
                  />
                  <div className="mt-2 flex gap-2">
                    <Button size="sm" type="submit">
                      Save
                    </Button>
                    <Button
                      size="sm"
                      type="button"
                      variant="ghost"
                      onClick={() => setEditing(null)}
                    >
                      Cancel
                    </Button>
                  </div>
                </form>
              ) : (
                <>
                  <button
                    className="block w-full text-left"
                    onClick={() => open(chat.id)}
                  >
                    <span className="line-clamp-2 text-[13px] leading-5">
                      {chat.title}
                    </span>
                    <span className="mt-1 block text-[11px] text-slate-500">
                      {chat.messageCount} messages
                      {chat.active ? " · Responding" : ""}
                    </span>
                  </button>
                  <div className="mt-1 flex justify-end">
                    <IconButton
                      label={`Rename ${chat.title}`}
                      onClick={() => {
                        setEditing(chat.id)
                        setRename(chat.title)
                      }}
                    >
                      <PencilSimpleIcon />
                    </IconButton>
                    <IconButton
                      label={`Delete ${chat.title}`}
                      onClick={() => setDeleting(chat.id)}
                    >
                      <TrashIcon />
                    </IconButton>
                  </div>
                  {deleting === chat.id && (
                    <div className="mt-2 text-xs">
                      <p>Delete this chat permanently?</p>
                      <div className="mt-2 flex gap-2">
                        <Button
                          size="sm"
                          disabled={deleteState.waiting}
                          onClick={() => {
                            void remove({ conversationId: chat.id }).then(
                              () => {
                                setDeleting(null)
                                if (selected === chat.id) open(null)
                              },
                              () => {}
                            )
                          }}
                        >
                          Delete
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setDeleting(null)}
                        >
                          Cancel
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      ))}
      {
        <div className="flex justify-between gap-2">
          {offset > 0 && (
            <Button
              variant="ghost"
              onClick={() => setOffset(Math.max(0, offset - limit))}
            >
              Newer
            </Button>
          )}
          {Result.isSuccess(history) && history.value.hasMore && (
            <Button variant="ghost" onClick={() => setOffset(offset + limit)}>
              Older
            </Button>
          )}
        </div>
      }
    </>
  )
}
