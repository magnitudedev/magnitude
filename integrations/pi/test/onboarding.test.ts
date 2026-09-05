import * as NodeFileSystem from "@effect/platform-node/NodeFileSystem"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect } from "effect"
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent"
import { describe, expect, it, vi } from "vitest"
import { canOfferSetup, claimSetupOffer, offerSetup, registerMagnitudeOnboarding, SETUP_PROMPT, SETUP_QUESTION, SETUP_REMINDER } from "../extensions/onboarding"

const context = (overrides: Record<string, unknown> = {}) => ({
  mode: "tui", isIdle: () => true, hasPendingMessages: () => false,
  ui: { getEditorText: () => "", notify: vi.fn() },
  sessionManager: { getEntries: () => [] }, modelRegistry: { getAll: () => [] },
  ...overrides,
}) as unknown as ExtensionContext

describe("Magnitude setup", () => {
  it.each([true, false, undefined])("records the decision and delivers only the intended message (%s)", async (answer) => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped()
      const confirm = vi.fn(async () => answer)
      const sendUserMessage = vi.fn()
      const sendMessage = vi.fn()
      const pi = { sendUserMessage, sendMessage } as unknown as ExtensionAPI
      const ctx = context({ ui: { confirm } })
      const signal = new AbortController().signal
      yield* offerSetup(pi, ctx, directory, signal)
      yield* offerSetup(pi, ctx, directory, signal)
      expect(confirm).toHaveBeenCalledExactlyOnceWith(SETUP_QUESTION, "", { signal })
      if (answer) {
        expect(sendUserMessage).toHaveBeenCalledExactlyOnceWith(SETUP_PROMPT)
        expect(sendMessage).not.toHaveBeenCalled()
      } else {
        expect(sendUserMessage).not.toHaveBeenCalled()
        expect(sendMessage).toHaveBeenCalledExactlyOnceWith({ customType: "magnitude-setup", content: SETUP_REMINDER, display: true })
      }
    })).pipe(Effect.provide(NodeFileSystem.layer)))
  })
  it("does not send onboarding when the extension is disposed during the dialog", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped()
      const abort = new AbortController()
      const sendUserMessage = vi.fn()
      const sendMessage = vi.fn()
      const ctx = context({ ui: { confirm: async () => { abort.abort(); return true } } })
      yield* offerSetup({ sendUserMessage, sendMessage } as unknown as ExtensionAPI, ctx, directory, abort.signal)
      expect(sendUserMessage).not.toHaveBeenCalled()
      expect(sendMessage).not.toHaveBeenCalled()
    })).pipe(Effect.provide(NodeFileSystem.layer)))
  })
  it("claims one offer across concurrent launches, independently per profile", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped()
      const results = yield* Effect.all(Array.from({ length: 12 }, () => claimSetupOffer(`${directory}/one`)), { concurrency: "unbounded" })
      expect(results.filter(Boolean)).toHaveLength(1)
      expect(yield* claimSetupOffer(`${directory}/one`)).toBe(false)
      expect(yield* claimSetupOffer(`${directory}/two`)).toBe(true)
    })).pipe(Effect.provide(NodeFileSystem.layer)))
  })
  it.each(["rpc", "json", "print"])("never offers in %s mode", (mode) => {
    expect(canOfferSetup(context({ mode }), [])).toBe(false)
  })
  it.each([
    { isIdle: () => false },
    { hasPendingMessages: () => true },
    { ui: { getEditorText: () => "unfinished input" } },
    { sessionManager: { getEntries: () => [{ type: "message" }] } },
    { modelRegistry: { getAll: () => [{ provider: "magnitude" }] } },
  ])("leaves existing work/configuration alone (%j)", (overrides) => {
    expect(canOfferSetup(context(overrides), [])).toBe(false)
  })
  it.each([["do something"], ["@instructions.md"], ["--model", "some-model", "work"]])("defers command-line work (%j)", (...argv) => {
    expect(canOfferSetup(context(), argv)).toBe(false)
  })
  it("allows an empty new interactive session with CLI options", () => {
    expect(canOfferSetup(context(), ["--model", "some-model", "--no-context-files"])).toBe(true)
  })
  it("manual setup sends the exact prompt once and never invokes a CLI", async () => {
    const registerCommand = vi.fn()
    const sendUserMessage = vi.fn()
    const dispose = registerMagnitudeOnboarding({ registerCommand, sendUserMessage, on: vi.fn() } as unknown as ExtensionAPI)
    expect(registerCommand.mock.calls[0]![0]).toBe("magnitude-setup")
    await registerCommand.mock.calls[0]![1].handler("", context())
    expect(sendUserMessage).toHaveBeenCalledExactlyOnceWith(SETUP_PROMPT)
    await dispose()
  })
  it("manual setup does not interrupt an active task", async () => {
    const registerCommand = vi.fn()
    const sendUserMessage = vi.fn()
    const dispose = registerMagnitudeOnboarding({ registerCommand, sendUserMessage, on: vi.fn() } as unknown as ExtensionAPI)
    await registerCommand.mock.calls[0]![1].handler("", context({ isIdle: () => false }))
    expect(sendUserMessage).not.toHaveBeenCalled()
    await dispose()
  })
  it("uses a bundled skill only when no Magnitude skill is already loaded", async () => {
    const handlers = new Map<string, (...args: any[]) => any>()
    const getCommands = vi.fn(() => [] as any[])
    const dispose = registerMagnitudeOnboarding({ registerCommand: vi.fn(), getCommands, on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, handler) } as unknown as ExtensionAPI)
    const discover = handlers.get("resources_discover")!
    expect(discover().skillPaths[0]).toMatch(/\/skills\/magnitude\/SKILL.md$/)
    getCommands.mockReturnValue([{ source: "skill", name: "skill:magnitude" }])
    expect(discover()).toBeUndefined()
    getCommands.mockReturnValue([{ source: "extension", name: "magnitude-setup" }])
    expect(discover().skillPaths).toHaveLength(1)
    await dispose()
  })
  it("respects an explicit --no-skills request", async () => {
    const handlers = new Map<string, (...args: any[]) => any>()
    const dispose = registerMagnitudeOnboarding({ registerCommand: vi.fn(), getCommands: () => [], on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, handler) } as unknown as ExtensionAPI)
    const argv = process.argv
    try {
      process.argv = ["node", "pi", "--no-skills"]
      expect(handlers.get("resources_discover")!()).toBeUndefined()
    } finally {
      process.argv = argv
      await dispose()
    }
  })
})
