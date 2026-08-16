import { EventEmitter } from "node:events"
import { Registry } from "@effect-atom/atom-react"
import { CliRenderEvents, type CliRenderer, type TerminalColors } from "@opentui/core"
import { Effect, Exit, Scope } from "effect"
import { describe, expect, it, vi } from "vitest"
import { terminalAppearanceAtom } from "../hooks/use-theme"
import { detectTerminalAppearance, installTerminalAppearanceRuntime } from "./terminal-appearance"

const colors = (foreground: string | null, background: string | null): TerminalColors => ({
  palette: [],
  defaultForeground: foreground,
  defaultBackground: background,
  cursorColor: null,
  mouseForeground: null,
  mouseBackground: null,
  tekForeground: null,
  tekBackground: null,
  highlightBackground: null,
  highlightForeground: null,
})

const renderer = (
  terminalColors: TerminalColors | Error,
  mode: "dark" | "light" | null,
): CliRenderer => ({
  getPalette: () => terminalColors instanceof Error
    ? Promise.reject(terminalColors)
    : Promise.resolve(terminalColors),
  waitForThemeMode: () => Promise.resolve(mode),
  capabilities: { rgb: true, ansi256: true, color_scheme_updates: true },
}) as unknown as CliRenderer

describe("terminal appearance detection", () => {
  it("treats the reported background as stronger evidence than a mode hint", async () => {
    const detected = await Effect.runPromise(detectTerminalAppearance(
      renderer(colors("#202020", "#fafafa"), "dark"),
      {},
    ))

    expect(detected.mode).toBe("light")
    expect(detected.defaultBackground).toBe("#fafafa")
  })

  it("chooses the mode whose text has better contrast on a midtone background", async () => {
    const detected = await Effect.runPromise(detectTerminalAppearance(
      renderer(colors("#eeeeee", "#808080"), "dark"),
      {},
    ))

    expect(detected.mode).toBe("light")
  })

  it("uses standard environment hints when color queries and renderer mode are unavailable", async () => {
    const detected = await Effect.runPromise(detectTerminalAppearance(
      renderer(new Error("unsupported"), null),
      { COLORFGBG: "15;0" },
    ))

    expect(detected.mode).toBe("dark")
  })

  it("uses the matching light fallback colors when environment detection is the only evidence", async () => {
    const detected = await Effect.runPromise(detectTerminalAppearance(
      renderer(new Error("unsupported"), null),
      { TERM_BACKGROUND: "light" },
    ))

    expect(detected.mode).toBe("light")
    expect(detected.defaultBackground).toBe("#ffffff")
  })

  it("recognizes ANSI bright black as a dark COLORFGBG background", async () => {
    const detected = await Effect.runPromise(detectTerminalAppearance(
      renderer(new Error("unsupported"), null),
      { COLORFGBG: "15;8" },
    ))

    expect(detected.mode).toBe("dark")
  })

  it("preserves the last known colors after a failed refresh", async () => {
    const previous = {
      mode: "light" as const,
      defaultBackground: "#f4f1e8",
    }
    const detected = await Effect.runPromise(detectTerminalAppearance(
      renderer(new Error("timed out"), null),
      {},
      previous,
    ))

    expect(detected.mode).toBe("light")
    expect(detected.defaultBackground).toBe(previous.defaultBackground)
  })

  it("uses a matching fallback background when a mode update outlives palette detection", async () => {
    const detected = await Effect.runPromise(detectTerminalAppearance(
      renderer(new Error("timed out"), "light"),
      {},
      { mode: "dark", defaultBackground: "#101010" },
    ))

    expect(detected).toEqual({ mode: "light", defaultBackground: "#ffffff" })
  })

  it("releases every renderer listener with its Effect scope", async () => {
    class RuntimeRenderer extends EventEmitter {
      readonly capabilities = { rgb: true, ansi256: true, color_scheme_updates: true }
      getPalette = () => Promise.resolve(colors("#eeeeee", "#101010"))
      waitForThemeMode = () => Promise.resolve<"dark">("dark")
      clearPaletteCache = () => undefined
    }

    const runtimeRenderer = new RuntimeRenderer()
    const scope = await Effect.runPromise(Scope.make())
    await Effect.runPromise(installTerminalAppearanceRuntime(
      runtimeRenderer as unknown as CliRenderer,
      Registry.make(),
      {},
    ).pipe(Effect.provideService(Scope.Scope, scope)))

    expect(runtimeRenderer.listenerCount(CliRenderEvents.THEME_MODE)).toBe(1)
    expect(runtimeRenderer.listenerCount(CliRenderEvents.PALETTE)).toBe(1)
    expect(runtimeRenderer.listenerCount(CliRenderEvents.CAPABILITIES)).toBe(1)
    expect(runtimeRenderer.listenerCount(CliRenderEvents.FOCUS)).toBe(1)

    await Effect.runPromise(Scope.close(scope, Exit.void))

    expect(runtimeRenderer.eventNames()).toEqual([])
  })

  it("applies a live theme-mode update without clearing OpenTUI's cache again", async () => {
    class RuntimeRenderer extends EventEmitter {
      readonly capabilities = { rgb: true, ansi256: true, color_scheme_updates: true }
      currentColors = colors("#eeeeee", "#101010")
      currentMode: "dark" | "light" = "dark"
      clearCount = 0
      getPalette = () => Promise.resolve(this.currentColors)
      waitForThemeMode = () => Promise.resolve(this.currentMode)
      clearPaletteCache = () => { this.clearCount += 1 }
    }

    const runtimeRenderer = new RuntimeRenderer()
    const registry = Registry.make()
    registry.set(terminalAppearanceAtom, { mode: "dark", defaultBackground: "#101010" })
    const scope = await Effect.runPromise(Scope.make())
    await Effect.runPromise(installTerminalAppearanceRuntime(
      runtimeRenderer as unknown as CliRenderer,
      registry,
      {},
    ).pipe(Effect.provideService(Scope.Scope, scope)))

    runtimeRenderer.currentColors = colors("#202020", "#fafafa")
    runtimeRenderer.currentMode = "light"
    runtimeRenderer.emit(CliRenderEvents.THEME_MODE, "light")

    await vi.waitFor(() => {
      expect(registry.get(terminalAppearanceAtom)).toEqual({
        mode: "light",
        defaultBackground: "#fafafa",
      })
    })
    expect(runtimeRenderer.clearCount).toBe(0)

    await Effect.runPromise(Scope.close(scope, Exit.void))
  })

  it("clears a cached failed palette query when terminal capabilities arrive", async () => {
    class RuntimeRenderer extends EventEmitter {
      readonly capabilities = { rgb: true, ansi256: true, color_scheme_updates: true }
      clearCount = 0
      getPalette = () => Promise.resolve(colors("#eeeeee", "#101010"))
      waitForThemeMode = () => Promise.resolve<"dark">("dark")
      clearPaletteCache = () => { this.clearCount += 1 }
    }

    const runtimeRenderer = new RuntimeRenderer()
    const scope = await Effect.runPromise(Scope.make())
    await Effect.runPromise(installTerminalAppearanceRuntime(
      runtimeRenderer as unknown as CliRenderer,
      Registry.make(),
      {},
    ).pipe(Effect.provideService(Scope.Scope, scope)))

    runtimeRenderer.emit(CliRenderEvents.CAPABILITIES, runtimeRenderer.capabilities)
    await vi.waitFor(() => expect(runtimeRenderer.clearCount).toBe(1))

    await Effect.runPromise(Scope.close(scope, Exit.void))
  })
})
