import { Registry } from "@effect-atom/atom-react"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import {
  CliRenderEvents,
  type CliRenderer,
  type TerminalColors,
  type ThemeMode as OpenTuiThemeMode,
} from "@opentui/core"
import {
  defaultGlobalStorageRoot,
  readStructuredFile,
  writeStructuredFileAtomic,
} from "@magnitudedev/storage"
import { Chunk, Effect, Option, Queue, Runtime, Schema, Scope } from "effect"
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
})

const sameAppearance = (left: TerminalAppearance, right: TerminalAppearance): boolean =>
  left.mode === right.mode
  && left.defaultBackground === right.defaultBackground

/**
 * The last detected appearance, persisted across runs so the first frame never
 * waits on detection: paint with the last answer, correct live if detection
 * disagrees.
 */
const TerminalAppearanceSchema = Schema.Struct({
  mode: Schema.Literal("dark", "light"),
  defaultBackground: Schema.NullOr(Schema.String),
})

const persistedAppearancePath = (path: Path.Path): string =>
  path.join(defaultGlobalStorageRoot(), "appearance.json")

export const readPersistedTerminalAppearance: Effect.Effect<
  Option.Option<TerminalAppearance>,
  never,
  FileSystem.FileSystem | Path.Path
> = Path.Path.pipe(
  Effect.flatMap((path) => readStructuredFile(
    persistedAppearancePath(path),
    TerminalAppearanceSchema,
  )),
  Effect.map((result) => result._tag === "Present"
    ? Option.some<TerminalAppearance>(result.value)
    : Option.none<TerminalAppearance>()),
  Effect.catchAll(() => Effect.succeed(Option.none<TerminalAppearance>())),
)

export const persistTerminalAppearance = (
  appearance: TerminalAppearance,
): Effect.Effect<void, never, FileSystem.FileSystem | Path.Path> => Path.Path.pipe(
  Effect.flatMap((path) => writeStructuredFileAtomic(
    persistedAppearancePath(path),
    TerminalAppearanceSchema,
    appearance,
    { mode: 0o600 },
  )),
  Effect.catchAll(() => Effect.void),
)

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
