import { renderToStaticMarkup } from "react-dom/server"
import { Atom, RegistryProvider, Result } from "@effect-atom/atom-react"
import { DateTime } from "effect"
import { describe, expect, it } from "vitest"
import { AgentClientProvider, selectedChat, type AgentClient } from "@magnitudedev/client-common"
import { SessionId, type ChatPage } from "@magnitudedev/sdk"
import type { MainPage } from "@magnitudedev/client-common"
import { ChatTabStatus, SidebarChats } from "./sidebar-list"

const time = { created: DateTime.unsafeMake(0), updated: DateTime.unsafeMake(0) }
const chat = (id: string, title: string, running = false) => ({ id: SessionId.make(id), title, time, messageCount: 2, running })

const render = (page: ChatPage, selected: string | null = null, shown: MainPage = "chat", part: "list" | "tab" = "list") => {
  const client = {
    Chat: {
      ListChats: () => Atom.make({ result: Result.success(page) }),
      RenameChat: Atom.make(Result.initial()),
      RemoveChat: Atom.make(Result.initial()),
    },
  } as unknown as AgentClient
  return renderToStaticMarkup(
    <RegistryProvider initialValues={[[selectedChat, selected === null ? null : SessionId.make(selected)]]}>
      <AgentClientProvider tag={client}>{part === "list" ? <SidebarChats page={shown} navigate={() => {}} /> : <ChatTabStatus page={shown} />}</AgentClientProvider>
    </RegistryProvider>
  )
}

describe("sidebar chat list", () => {
  it("lists the recent chats and marks the open one", () => {
    const html = render({ chats: [chat("a", "Plan the trip", true), chat("b", "Summarize notes")], hasMore: false }, "b")
    expect(html).toContain("Plan the trip")
    expect(html).toContain("Summarize notes")
    expect(html).toContain('aria-label="Options for Plan the trip"')
    expect(html.match(/aria-current="page"/g)).toHaveLength(1)
    expect(html).toMatch(/data-current="true"[^>]*>.*Summarize notes/)
  })

  it("puts a spinner in each chat's status slot while its response is in progress, and leaves the slot empty otherwise", () => {
    const html = render({ chats: [chat("a", "Plan the trip", true), chat("b", "Summarize notes")], hasMore: false })
    expect(html.match(/aria-label="In progress"/g)).toHaveLength(1)
    expect(html).toMatch(/Plan the trip<\/span><span class="chat-list-status"><svg[^>]*aria-label="In progress"/)
    expect(html).toContain('Summarize notes</span><span class="chat-list-status"></span>')
  })

  it("says so when there are no chats", () => {
    expect(render({ chats: [], hasMore: false })).toContain("No chats yet.")
  })

  it("shows nothing under Chat, neither chats nor that there are none, while Models or Connections is the page", () => {
    for (const shown of ["models", "connections"] as const) {
      expect(render({ chats: [chat("b", "Summarize notes")], hasMore: false }, "b", shown)).toBe("")
      expect(render({ chats: [], hasMore: false }, null, shown)).toBe("")
    }
  })
})

describe("the Chat tab's status", () => {
  const running = { chats: [chat("a", "Plan the trip", true)], hasMore: false }

  it("shows a spinner on Models and Connections while any chat's response is in progress", () => {
    for (const shown of ["models", "connections"] as const) expect(render(running, null, shown, "tab")).toContain('aria-label="In progress"')
  })

  it("shows nothing while no response is in progress, or on Chat, whose rows show it", () => {
    expect(render({ chats: [chat("b", "Summarize notes")], hasMore: false }, null, "models", "tab")).toBe("")
    expect(render(running, null, "chat", "tab")).toBe("")
  })
})
