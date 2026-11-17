import { renderToStaticMarkup } from "react-dom/server"
import { Atom, RegistryProvider, Result } from "@effect-atom/atom-react"
import { DateTime } from "effect"
import { describe, expect, it } from "vitest"
import { AgentClientProvider, selectedChat, type AgentClient } from "@magnitudedev/client-common"
import { SessionId, type ChatPage } from "@magnitudedev/sdk"
import { ChatSection } from "./sidebar-list"

const time = { created: DateTime.unsafeMake(0), updated: DateTime.unsafeMake(0) }
const chat = (id: string, title: string, running = false) => ({ id: SessionId.make(id), title, time, messageCount: 2, running })

const render = (page: ChatPage, selected: string | null = null) => {
  const client = {
    Chat: {
      ListChats: () => Atom.make({ result: Result.success(page) }),
      RenameChat: Atom.make(Result.initial()),
      RemoveChat: Atom.make(Result.initial()),
    },
  } as unknown as AgentClient
  return renderToStaticMarkup(
    <RegistryProvider initialValues={[[selectedChat, selected === null ? null : SessionId.make(selected)]]}>
      <AgentClientProvider tag={client}><ChatSection navigate={() => {}} /></AgentClientProvider>
    </RegistryProvider>
  )
}

describe("sidebar chat list", () => {
  it("lists chats under New chat, marks the open one, and shows no running spinner", () => {
    const html = render({ chats: [chat("a", "Plan the trip", true), chat("b", "Summarize notes")], hasMore: false }, "b")
    expect(html).toContain("New chat")
    expect(html).toContain("Plan the trip")
    expect(html).toContain("Summarize notes")
    expect(html).toContain('aria-label="Options for Plan the trip"')
    expect(html.match(/aria-current="page"/g)).toHaveLength(1)
    expect(html).toMatch(/data-current="true"[^>]*>.*Summarize notes/)
    expect(html).not.toContain("Working")
  })

  it("says how to start when there are no chats", () => {
    expect(render({ chats: [], hasMore: false })).toContain("No chats yet. Click New chat above to start one.")
  })
})
