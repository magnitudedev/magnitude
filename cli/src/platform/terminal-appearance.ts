import { Registry } from "@effect-atom/atom-react"
import {
  CliRenderEvents,
  createTerminalPalette,
  type CliRenderer,
  type TerminalColors,
  type ThemeMode as OpenTuiThemeMode,
} from "@opentui/core"
import { Chunk, Effect, Queue, Runtime, Scope } from "effect"
import { terminalAppearanceAtom } from "../hooks/use-theme"
import type { CliEnv } from "../types/env"
import type { TerminalAppearance, ThemeMode } from "../types/theme-system"
import { collectCliEnv } from "../utils/env"
import { fallbackTerminalAppearances, isLightBackground, parseHexColor } from "../utils/theme"

type AtomRegistry = ReturnType<typeof Registry.make>

const environmentMode = (env: CliEnv): ThemeMode | null => {
  const explicit = env.TERM_BACKGROUND?.trim().toLowerCase()
  if (explicit === "light" || explicit === "dark") return explicit

  const colorFgBg = env.COLORFGBG?.split(";").at(-1)?.trim()
  if (colorFgBg && /^\d+$/.test(colorFgBg)) {
    const backgroundIndex = Number.parseInt(colorFgBg, 10)
    return backgroundIndex === 8 || backgroundIndex < 7 ? "dark" : "light"
  }
  return null
}

const resolveMode = (
  colors: TerminalColors | null,
  rendererMode: OpenTuiThemeMode | null,
  env: CliEnv,
  fallbackMode: ThemeMode,
): ThemeMode => {
  const background = colors?.defaultBackground
  if (background && parseHexColor(background)) {
    return isLightBackground(background) ? "light" : "dark"
  }
  return rendererMode ?? environmentMode(env) ?? fallbackMode
}

const resolveAppearance = (
  colors: TerminalColors | null,
  rendererMode: OpenTuiThemeMode | null,
  env: CliEnv,
  previous?: TerminalAppearance,
): TerminalAppearance => {
  const mode = resolveMode(colors, rendererMode, env, previous?.mode ?? "dark")
  const fallback = fallbackTerminalAppearances[mode]
  const detectedBackground = colors?.defaultBackground
  const defaultBackground = detectedBackground && parseHexColor(detectedBackground)
    ? detectedBackground
    : previous?.mode === mode
      ? previous.defaultBackground
      : fallback.defaultBackground
  return {
    mode,
    defaultBackground,
  }
}

/**
 * Probes the terminal's colors before the renderer exists, with OpenTUI's own
 * standalone palette detector on the raw process streams. The support check
 * settles on the first reply and bounds a silent terminal at 100ms — the
 * probe's whole budget there; the full color queries run only once support is
 * proven, where they complete in reply roundtrips. Two boundary duties are
 * ours here because no renderer owns the terminal yet: raw mode (replies must
 * bypass canonical line buffering) and returning stdin to its non-flowing
 * state. Bytes typed during the probe window are consumed with the replies.
 */
const PROBE_SUPPORT_TIMEOUT_MILLIS = 100

export const probeTerminalAppearance = (
  env: CliEnv = collectCliEnv(),
): Effect.Effect<TerminalAppearance> => Effect.gen(function* () {
  if (process.stdin.isTTY !== true || process.stdout.isTTY !== true) {
    return resolveAppearance(null, null, env)
  }
  const colors = yield* Effect.acquireUseRelease(
    Effect.sync(() => {
      process.stdin.setRawMode(true)
      return createTerminalPalette({
        stdin: process.stdin,
        stdout: process.stdout,
      })
    }),
    (detector) => Effect.tryPromise(async () =>
      (await detector.detectOSCSupport(PROBE_SUPPORT_TIMEOUT_MILLIS))
        ? detector.detect({ size: 16 })
        : null,
    ).pipe(Effect.orElseSucceed((): TerminalColors | null => null)),
    (detector) => Effect.sync(() => {
      detector.cleanup()
      process.stdin.pause()
      process.stdin.setRawMode(false)
    }),
  )
  return resolveAppearance(colors, null, env)
})

const optionalPromise = <A>(evaluate: () => Promise<A>): Effect.Effect<A | null> =>
  Effect.tryPromise(evaluate).pipe(Effect.catchAll(() => Effect.succeed(null)))

export const detectTerminalAppearance = (
  renderer: CliRenderer,
  env: CliEnv = collectCliEnv(),
  previous?: TerminalAppearance,
): Effect.Effect<TerminalAppearance> => Effect.gen(function* () {
  const [colors, rendererMode] = yield* Effect.all([
    optionalPromise(() => renderer.getPalette({ size: 16, timeout: 700 })),
    optionalPromise(() => renderer.waitForThemeMode(700)),
  ], { concurrency: "unbounded" })
  return resolveAppearance(colors, rendererMode, env, previous)
})

const sameAppearance = (left: TerminalAppearance, right: TerminalAppearance): boolean =>
  left.mode === right.mode
  && left.defaultBackground === right.defaultBackground

export const installTerminalAppearanceRuntime = (
  renderer: CliRenderer,
  registry: AtomRegistry,
  env: CliEnv = collectCliEnv(),
): Effect.Effect<void, never, Scope.Scope> => Effect.gen(function* () {
  const refreshes = yield* Queue.unbounded<boolean>()
  const runtime = yield* Effect.runtime<never>()
  const run = Runtime.runFork(runtime)
  const requestRefresh = (clearCache: boolean): void => {
    run(Queue.offer(refreshes, clearCache))
  }

  // OpenTUI clears its palette cache before emitting THEME_MODE. Clearing it
  // again here can invalidate OpenTUI's in-flight palette publication.
  const onThemeMode = (): void => requestRefresh(false)
  const onPalette = (): void => requestRefresh(false)
  const onCapabilities = (): void => requestRefresh(true)
  const onFocus = (): void => {
    if (!renderer.capabilities?.color_scheme_updates) requestRefresh(true)
  }
  yield* Effect.acquireRelease(
    Effect.sync(() => {
      renderer.on(CliRenderEvents.THEME_MODE, onThemeMode)
      renderer.on(CliRenderEvents.PALETTE, onPalette)
      renderer.on(CliRenderEvents.CAPABILITIES, onCapabilities)
      renderer.on(CliRenderEvents.FOCUS, onFocus)
    }),
    () => Effect.sync(() => {
      renderer.off(CliRenderEvents.THEME_MODE, onThemeMode)
      renderer.off(CliRenderEvents.PALETTE, onPalette)
      renderer.off(CliRenderEvents.CAPABILITIES, onCapabilities)
      renderer.off(CliRenderEvents.FOCUS, onFocus)
    }),
  )

  yield* Effect.forever(Effect.gen(function* () {
    const firstRefresh = yield* Queue.take(refreshes)
    const pendingRefreshes = yield* Queue.takeAll(refreshes)
    const clearCache = firstRefresh || Chunk.some(pendingRefreshes, Boolean)
    if (clearCache) renderer.clearPaletteCache()
    const current = registry.get(terminalAppearanceAtom)
    const appearance = yield* detectTerminalAppearance(renderer, env, current)
    if (!sameAppearance(current, appearance)) registry.set(terminalAppearanceAtom, appearance)
  })).pipe(Effect.forkScoped)
}).pipe(Effect.asVoid)
