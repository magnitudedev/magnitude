import type { ExtensionAPI } from "@earendil-works/pi-coding-agent"
import { describe, expect, it, vi } from "vitest"
import magnitudeExtension from "../extensions/magnitude"

describe("Magnitude Pi extension", () => {
  it.each([false, true])("uses the real Pi parser's semantic outcome (error=%s)", async (failed) => {
    const events = new Map<string, (...args: any[]) => any>()
    const registerProvider = vi.fn()
    magnitudeExtension({ registerProvider, registerCommand: vi.fn(), on: (name: string, handler: (...args: any[]) => any) => events.set(name, handler) } as unknown as ExtensionAPI)
    const ui = { setWidget: vi.fn(), setWorkingMessage: vi.fn() }
    await events.get("session_start")!({}, { ui })
    const model = { id: "local", name: "Model", api: "openai-completions", provider: "magnitude", baseUrl: "http://localhost/v1", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 10000, maxTokens: 100 }
    events.get("agent_start")!({}, { model })
    const chunk = { id: "test", object: "chat.completion.chunk", created: 1, model: "local", choices: [{ index: 0, delta: { role: "assistant", content: "hello" }, finish_reason: null }], timings: { prompt_ms: 1, time_to_first_token_ms: 2, predicted_n: 1, predicted_ms: 10, predicted_per_second: 100 } }
    const wire = `data: ${JSON.stringify(chunk)}\n\n` + (failed
      ? 'data: {"error":{"message":"deliberate SSE failure","type":"server_error"}}\n\n'
      : 'data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"local","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n') + 'data: [DONE]\n\n'
    const upstream = Object.assign(vi.fn(async () => new Response(wire, { headers: { "content-type": "text/event-stream" } })), { preconnect: vi.fn() })
    const stream = registerProvider.mock.calls[0]![1].streamSimple(model, { messages: [{ role: "user", content: "hi", timestamp: 1 }] }, { apiKey: "local", fetch: upstream })
    const result = await stream.result()
    expect(result.stopReason).toBe(failed ? "error" : "stop")
    events.get("agent_settled")!()
    if (!failed) await vi.waitFor(() => expect(ui.setWidget.mock.calls.some((call) => typeof call[1] === "function")).toBe(true))
    else {
      await new Promise((resolve) => setTimeout(resolve, 20))
      expect(ui.setWidget.mock.calls.some((call) => typeof call[1] === "function")).toBe(false)
    }
    await events.get("session_shutdown")!()
  })
  it("registers an OpenAI Completions provider, commands, and tracker lifecycle", async () => {
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
    expect(commands).toEqual(["load-model", "stop-model", "magnitude-setup"])
    expect([...events.keys()]).toEqual([
      "resources_discover",
      "session_start",
      "model_select",
      "agent_start",
      "agent_settled",
      "session_shutdown",
    ])

    const setWidget = vi.fn()
    const setWorkingMessage = vi.fn()
    await events.get("session_start")?.({}, { ui: { setWidget, setWorkingMessage } })
    events.get("agent_start")?.({}, { model: { provider: "magnitude", name: "Magnitude Model" } })
    events.get("agent_settled")?.()
    events.get("model_select")?.({ model: { provider: "openai" } })
    expect(setWorkingMessage).toHaveBeenLastCalledWith()
    expect(setWidget).toHaveBeenLastCalledWith("magnitude-inference-summary", undefined)
    await events.get("session_shutdown")?.()
  })
})
