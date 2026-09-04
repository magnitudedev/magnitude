import type { ExtensionAPI } from "@earendil-works/pi-coding-agent"
import { describe, expect, it, vi } from "vitest"
import magnitudeExtension from "../extensions/magnitude"

describe("Magnitude Pi extension", () => {
  it("registers an OpenAI Completions provider, commands, and tracker lifecycle", () => {
    const events = new Map<string, (...args: unknown[]) => void>()
    const commands: string[] = []
    const registerProvider = vi.fn()
    const pi = {
      registerProvider,
      registerCommand: (name: string) => commands.push(name),
      on: (name: string, handler: (...args: unknown[]) => void) => events.set(name, handler),
    } as unknown as ExtensionAPI

    magnitudeExtension(pi)

    expect(registerProvider).toHaveBeenCalledWith("magnitude", expect.objectContaining({
      api: "openai-completions",
      streamSimple: expect.any(Function),
    }))
    expect(commands).toEqual(["load-model", "stop-model"])
    expect([...events.keys()]).toEqual([
      "session_start",
      "model_select",
      "agent_start",
      "agent_settled",
      "session_shutdown",
    ])

    const setWidget = vi.fn()
    const setWorkingMessage = vi.fn()
    events.get("session_start")?.({}, { ui: { setWidget, setWorkingMessage } })
    events.get("agent_start")?.({}, { model: { provider: "magnitude", name: "Magnitude Model" } })
    events.get("agent_settled")?.()
    events.get("model_select")?.({ model: { provider: "openai" } })
    expect(setWorkingMessage).toHaveBeenLastCalledWith()
    expect(setWidget).toHaveBeenLastCalledWith("magnitude-inference-summary", undefined)
    events.get("session_shutdown")?.()
  })
})
