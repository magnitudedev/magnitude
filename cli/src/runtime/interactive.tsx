import { resolve } from "path"
import type { Root } from "@opentui/react"
import type { CliRenderer } from "@opentui/core"
import { createCliRenderer } from "@opentui/core"
import { createRoot } from "@opentui/react"
import {
  Atom,
  Registry,
  RegistryContext,
  scheduleTask,
} from "@effect-atom/atom-react"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as Path from "@effect/platform/Path"
import {
  createAgentClient,
  deriveCliExitNotice,
  onboardingModelSetupViewAtom,
  pushNotificationAtom,
  stopDisplayViewController,
} from "@magnitudedev/client-common"
import type { UpdateAction } from "@magnitudedev/release"
import { logger } from "@magnitudedev/logger"
import { acnInstallationPresent, SDK_ACN_TARGET } from "@magnitudedev/sdk"
import {
  Array as Arr,
  Deferred,
  Effect,
  Fiber,
  Option,
  Runtime,
  Schema,
  Scope,
  Stream,
} from "effect"
import type { SessionOptions } from "@magnitudedev/sdk"
import type { CliAppProps, SessionStart } from "../app"
import type { AuthSource } from "../state/cli-atoms"
import { getLastSessionId } from "../state/last-session"
import { CLI_VERSION } from "../version"
import { makeCliEffectLoggingLayer } from "../platform/effect-logger"
import {
  installTerminalAppearanceRuntime,
  probeTerminalAppearance,
} from "../platform/terminal-appearance"
import {
  makeTerminalPlatform,
  type TerminalPlatformRuntime,
} from "../platform/terminal"
import {
  makeProcessExitSource,
  restoreTerminalState,
  type ProcessExitRequest,
} from "../platform/process-exit"
import { terminalAppearanceAtom } from "../hooks/use-theme"
import type { UpdatePromptOutcome } from "../features/update/prompt"
import { executeUpdate } from "../features/update/execute"
import { CliUpdater, makeCliUpdater } from "../features/update/updater"
import {
  CliStartupRoot,
  makeCliRootStateAtom,
  type CliRootState,
} from "./root"

export class CliRendererAcquisitionFailed extends Schema.TaggedError<CliRendererAcquisitionFailed>()(
  "CliRendererAcquisitionFailed",
  { reason: Schema.String },
) {}

export interface InteractiveLaunchOptions {
  readonly debug: boolean
  readonly setup: boolean
  readonly developmentBuild: boolean
  readonly sessionStart: SessionStart
  readonly initialPrompt: string | undefined
  readonly goal: string | undefined
  readonly envAuth: AuthSource
  readonly sessionOptions: SessionOptions
}

type InteractiveSessionResult =
  | {
      readonly _tag: "UpdateRequested"
      readonly action: UpdateAction
    }
  | {
      readonly _tag: "Exit"
      readonly code: number
      readonly notices: ReadonlyArray<string>
      readonly fatal: Option.Option<{
        readonly label: string
        readonly message: string
        readonly stack: Option.Option<string>
      }>
    }

type ExitRace<A> =
  | { readonly _tag: "Value"; readonly value: A }
  | { readonly _tag: "Exit"; readonly request: ProcessExitRequest }

interface RootCallbacks {
  readonly onUpdateSelect: (outcome: UpdatePromptOutcome) => void
  readonly onDaemonRetry: () => void
  readonly onDaemonQuit: () => void
}

const acquireRegistry = Effect.acquireRelease(
  Effect.sync(() => Registry.make({
    scheduleTask,
    defaultIdleTTL: 5_000,
  })),
  (registry) => Effect.sync(() => registry.dispose()),
)

const acquireRenderer: Effect.Effect<
  CliRenderer,
  CliRendererAcquisitionFailed,
  Scope.Scope
> = Effect.acquireRelease(
  Effect.tryPromise({
    try: () => createCliRenderer({ exitOnCtrlC: false }),
    catch: (error) => new CliRendererAcquisitionFailed({
      reason: String(error),
    }),
  }),
  (renderer) => Effect.sync(() => {
    restoreTerminalState()
    try {
      renderer.destroy()
    } catch {
      // Terminal restoration must remain best-effort during finalization.
    }
  }),
)

const acquireRoot = (renderer: CliRenderer) => Effect.acquireRelease(
  Effect.sync(() => createRoot(renderer)),
  (root) => Effect.sync(() => root.unmount()),
)

const renderRoot = (
  root: Root,
  registry: ReturnType<typeof Registry.make>,
  stateAtom: Atom.Atom<CliRootState>,
  callbacks: RootCallbacks,
): Effect.Effect<void> => Effect.sync(() => {
  root.render(
    <RegistryContext.Provider value={registry}>
      <CliStartupRoot
        stateAtom={stateAtom}
        onUpdateSelect={callbacks.onUpdateSelect}
        onDaemonRetry={callbacks.onDaemonRetry}
        onDaemonQuit={callbacks.onDaemonQuit}
      />
    </RegistryContext.Provider>,
  )
})

const raceExit = <A, E, R>(
  effect: Effect.Effect<A, E, R>,
  exit: Effect.Effect<ProcessExitRequest>,
): Effect.Effect<ExitRace<A>, E, R> => Effect.race(
  effect.pipe(Effect.map((value): ExitRace<A> => ({ _tag: "Value", value }))),
  exit.pipe(Effect.map((request): ExitRace<A> => ({ _tag: "Exit", request }))),
)

const fatalResult = (
  request: Extract<ProcessExitRequest, { readonly _tag: "Fatal" }>,
): InteractiveSessionResult => {
  const isError = request.error instanceof Error
  return {
    _tag: "Exit",
    code: 1,
    notices: [],
    fatal: Option.some({
      label: request.label,
      message: isError ? request.error.message : String(request.error),
      stack: Option.fromNullable(isError ? request.error.stack : undefined),
    }),
  }
}

const preApplicationExit = (
  request: ProcessExitRequest,
): InteractiveSessionResult => request._tag === "Fatal"
  ? fatalResult(request)
  : { _tag: "Exit", code: 0, notices: [], fatal: Option.none() }

const closeApplication = (
  terminal: TerminalPlatformRuntime,
  request: ProcessExitRequest,
): Effect.Effect<InteractiveSessionResult> => request._tag === "Fatal"
  ? Effect.succeed(fatalResult(request))
  : terminal.close.pipe(
      Effect.map((observation) => {
        const notices: string[] = []
        const modelNotice = Option.getOrUndefined(deriveCliExitNotice(observation))
        if (modelNotice) notices.push(modelNotice)
        const activeSessionId = getLastSessionId()
        if (activeSessionId) {
          notices.push(`Resume this session with:\nmagnitude --resume ${activeSessionId}`)
        }
        stopDisplayViewController()
        return {
          _tag: "Exit" as const,
          code: 0,
          notices,
          fatal: Option.none(),
        }
      }),
    )

const runInteractiveSession = (
  options: InteractiveLaunchOptions,
): Effect.Effect<
  InteractiveSessionResult,
  CliRendererAcquisitionFailed,
  CliUpdater | FileSystem.FileSystem | Path.Path | Scope.Scope
> => Effect.gen(function* () {
  const updater = yield* CliUpdater
  const registry = yield* acquireRegistry
  const effectLoggingLayer = makeCliEffectLoggingLayer({
    debug: options.debug,
    publishNotification: (notification) => {
      registry.set(pushNotificationAtom, notification)
    },
  })
  const processExit = yield* makeProcessExitSource
  const runtime = yield* Effect.runtime<never>()

  // The appearance probe starts with the session — concurrent with discovery
  // and daemon work — and self-terminates within one terminal roundtrip
  // (fence) or its 100ms ceiling, so its answer is in long before anything
  // paints. It owns terminal input until then; the renderer is only created
  // after it is joined.
  const appearanceProbe = yield* Effect.forkScoped(probeTerminalAppearance())

  let updateSelectionHandler: (outcome: UpdatePromptOutcome) => void = () => {}
  let daemonRetryHandler: () => void = () => {}
  const callbacks: RootCallbacks = {
    onUpdateSelect: (outcome) => { updateSelectionHandler(outcome) },
    onDaemonRetry: () => { daemonRetryHandler() },
    onDaemonQuit: () => { process.kill(process.pid, "SIGINT") },
  }

  // The renderer (and with it the alternate screen) is acquired at the first
  // moment a state must render, never earlier; the root is created once.
  const mountedAtom: { current: Atom.Writable<CliRootState> | null } = {
    current: null,
  }
  const mountPresentation = (initial: CliRootState) => Effect.gen(function* () {
    registry.set(terminalAppearanceAtom, yield* Fiber.join(appearanceProbe))
    const renderer = yield* acquireRenderer
    yield* installTerminalAppearanceRuntime(renderer, registry)
    const stateAtom = makeCliRootStateAtom(initial)
    const root = yield* acquireRoot(renderer)
    yield* renderRoot(root, registry, stateAtom, callbacks)
    mountedAtom.current = stateAtom
  })
  const present = (
    state: CliRootState,
  ): Effect.Effect<
    void,
    CliRendererAcquisitionFailed,
    FileSystem.FileSystem | Path.Path | Scope.Scope
  > => Effect.suspend(() => {
    const stateAtom = mountedAtom.current
    return stateAtom !== null
      ? Effect.sync(() => registry.set(stateAtom, state))
      : mountPresentation(state)
  })

  // Update interaction happens only on plain interactive launches with a
  // known package-manager action; discovery itself still runs and caches.
  const promptAction: Option.Option<UpdateAction> =
    (options.initialPrompt?.length ?? 0) === 0
    && process.stdin.isTTY === true
    && process.stdout.isTTY === true
      ? updater.updateAction
      : Option.none()
  const discovery = yield* updater.discover.pipe(Effect.provide(effectLoggingLayer))
  let freshPending = true
  const declinedVersions = new Set<string>()
  const offerable = (latest: Option.Option<string>): Option.Option<string> =>
    Option.filter(latest, (version) => !declinedVersions.has(version))

  const presentUpdatePrompt = (latestVersion: string, action: UpdateAction) =>
    Effect.gen(function* () {
      const selection = yield* Deferred.make<UpdatePromptOutcome>()
      updateSelectionHandler = (outcome) => {
        Deferred.unsafeDone(selection, Effect.succeed(outcome))
      }
      yield* present({
        _tag: "UpdatePrompt",
        currentVersion: CLI_VERSION,
        latestVersion,
        action,
      })
      const selected = yield* raceExit(Deferred.await(selection), processExit.await)
      if (selected._tag === "Exit") {
        return { _tag: "Exit", request: selected.request } as const
      }
      if (selected.value._tag === "Dismiss") {
        yield* updater.dismissVersion(latestVersion)
      }
      if (selected.value._tag === "Update") return { _tag: "Update" } as const
      declinedVersions.add(latestVersion)
      return { _tag: "Declined" } as const
    })

  if (Option.isSome(promptAction) && Option.isSome(discovery.known)) {
    const resolution = yield* presentUpdatePrompt(
      discovery.known.value,
      promptAction.value,
    )
    if (resolution._tag === "Exit") return preApplicationExit(resolution.request)
    if (resolution._tag === "Update") {
      return { _tag: "UpdateRequested", action: promptAction.value }
    }
  }

  // The installation gate: with no installed daemon build this launch is about
  // to download one, so it awaits the version check first — an offer must
  // prompt before any download of the version it would replace. Installation
  // needs the network regardless, so the wait costs nothing real.
  if (Option.isSome(promptAction)
    && !(yield* acnInstallationPresent(SDK_ACN_TARGET.identity))) {
    const answer = yield* raceExit(discovery.fresh, processExit.await)
    if (answer._tag === "Exit") return preApplicationExit(answer.request)
    freshPending = false
    const offer = offerable(answer.value)
    if (Option.isSome(offer)) {
      const resolution = yield* presentUpdatePrompt(offer.value, promptAction.value)
      if (resolution._tag === "Exit") return preApplicationExit(resolution.request)
      if (resolution._tag === "Update") {
        return { _tag: "UpdateRequested", action: promptAction.value }
      }
    }
  }

  const platformResult = yield* raceExit(
    makeTerminalPlatform({
      launchCommand: developmentLaunchCommand(options),
      debug: options.debug,
      effectLoggingLayer: Option.some(effectLoggingLayer),
    }),
    processExit.await,
  )
  if (platformResult._tag === "Exit") return preApplicationExit(platformResult.request)
  const terminal = platformResult.value
  daemonRetryHandler = () => {
    Runtime.runFork(runtime)(terminal.platform.acnStartup.retry.pipe(Effect.ignore))
  }

  const prepared = yield* raceExit(
    terminal.platform.acnStartup.prepare,
    processExit.await,
  )
  if (prepared._tag === "Exit") return preApplicationExit(prepared.request)

  if (prepared.value._tag !== "Ready") {
    yield* present({ _tag: "DaemonStartup", lifecycle: prepared.value })

    const driveToReady = terminal.platform.acnStartup.state.changes.pipe(
      Stream.tap((state) => state._tag === "Ready"
        ? Effect.void
        : present({ _tag: "DaemonStartup", lifecycle: state })),
      Stream.filter((state) => state._tag === "Ready"),
      Stream.runHead,
      Effect.asVoid,
    )

    type DaemonWaitOutcome =
      | { readonly _tag: "Ready" }
      | { readonly _tag: "FreshAnswer"; readonly latest: Option.Option<string> }
      | { readonly _tag: "Exit"; readonly request: ProcessExitRequest }

    // The daemon work races the discovery answer: an offer arriving before the
    // app commits presents the prompt, and daemon work continues behind it —
    // declining loses nothing, accepting interrupts old-version work by
    // closing the scope.
    let starting = true
    while (starting) {
      const waiters: Array<Effect.Effect<
        DaemonWaitOutcome,
        CliRendererAcquisitionFailed,
        FileSystem.FileSystem | Path.Path | Scope.Scope
      >> = [
        driveToReady.pipe(Effect.map((): DaemonWaitOutcome => ({ _tag: "Ready" }))),
        processExit.await.pipe(Effect.map(
          (request): DaemonWaitOutcome => ({ _tag: "Exit", request }),
        )),
      ]
      if (Option.isSome(promptAction) && freshPending) {
        waiters.push(discovery.fresh.pipe(Effect.map(
          (latest): DaemonWaitOutcome => ({ _tag: "FreshAnswer", latest }),
        )))
      }
      const outcome = yield* Effect.raceAll(waiters)
      if (outcome._tag === "Exit") return preApplicationExit(outcome.request)
      if (outcome._tag === "Ready") {
        starting = false
        continue
      }
      freshPending = false
      const offer = offerable(outcome.latest)
      if (Option.isNone(offer) || Option.isNone(promptAction)) continue
      const resolution = yield* presentUpdatePrompt(offer.value, promptAction.value)
      if (resolution._tag === "Exit") return preApplicationExit(resolution.request)
      if (resolution._tag === "Update") {
        return { _tag: "UpdateRequested", action: promptAction.value }
      }
      yield* present({
        _tag: "DaemonStartup",
        lifecycle: yield* terminal.platform.acnStartup.state.get,
      })
    }
  }

  const connected = yield* raceExit(Effect.gen(function* () {
    const agentClient = createAgentClient(terminal.platform.protocolLayer, {
      onboardingSetupInitiallyOpen: options.setup,
    })
    yield* Effect.exit(Registry.getResult(
      registry,
      onboardingModelSetupViewAtom(agentClient),
    ))
    return agentClient
  }), processExit.await)
  if (connected._tag === "Exit") return preApplicationExit(connected.request)

  const app: CliAppProps = {
    sessionStart: options.sessionStart,
    initialPrompt: options.initialPrompt,
    goal: options.goal,
    envAuth: options.envAuth,
    initialAcnLifecycle: yield* terminal.platform.acnStartup.state.get,
    sessionOptions: options.sessionOptions,
  }
  yield* present({
    _tag: "Application",
    platform: terminal.platform,
    agentClient: connected.value,
    app,
  })

  // A discovery answer arriving after the app committed surfaces as one
  // notification line; the prompt would interrupt real work now.
  if (Option.isSome(promptAction) && freshPending) {
    yield* Effect.forkScoped(discovery.fresh.pipe(
      Effect.flatMap((latest) => Option.match(offerable(latest), {
        onNone: () => Effect.void,
        onSome: (version) => Effect.sync(() => {
          registry.set(pushNotificationAtom, {
            message: `Update available: ${version} — restart or run \`magnitude update\``,
            priority: "notice",
            action: Option.none(),
            dismissAfterMilliseconds: 30_000,
          })
        }),
      })),
    ))
  }

  const request = yield* processExit.await
  return yield* closeApplication(terminal, request)
})

const writeSessionResult = (
  result: Extract<InteractiveSessionResult, { readonly _tag: "Exit" }>,
): Effect.Effect<void> => Effect.sync(() => {
  process.exitCode = result.code
  if (Option.isSome(result.fatal)) {
    const fatal = result.fatal.value
    logger.error({
      error: fatal.message,
      stack: Option.getOrUndefined(fatal.stack),
    }, fatal.label)
    process.stderr.write(`\n${fatal.label}: ${fatal.message}\n`)
    if (Option.isSome(fatal.stack)) process.stderr.write(`${fatal.stack.value}\n`)
  }
  if (result.notices.length > 0) {
    process.stdout.write(`\n${result.notices.join("\n\n")}\n`)
  }
})

export const runInteractiveCommand = (
  options: InteractiveLaunchOptions,
) => Effect.gen(function* () {
  const updater = yield* makeCliUpdater({
    currentVersion: CLI_VERSION,
    developmentBuild: options.developmentBuild,
  })
  const result = yield* Effect.scoped(runInteractiveSession(options).pipe(
    Effect.provideService(CliUpdater, updater),
  ))
  if (result._tag === "UpdateRequested") {
    yield* executeUpdate(updater, result.action, { relaunch: true })
  } else {
    yield* writeSessionResult(result)
  }
})

const developmentLaunchCommand = (
  options: InteractiveLaunchOptions,
): Option.Option<Arr.NonEmptyReadonlyArray<string>> => options.developmentBuild
  ? Option.some([
      "bun",
      resolve(import.meta.dir, "..", "..", "..", "packages", "acn", "src", "binary.ts"),
      "serve",
      ...(options.debug ? ["--debug"] : []),
    ])
  : Option.none()
