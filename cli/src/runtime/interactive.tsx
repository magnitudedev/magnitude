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
import {
  createAgentClient,
  deriveCliExitNotice,
  onboardingModelSetupViewAtom,
  pushNotificationAtom,
  stopDisplayViewController,
} from "@magnitudedev/client-common"
import {
  CliUpdater,
  makeCliUpdater,
  type UpdateAction,
} from "@magnitudedev/sdk"
import { logger } from "@magnitudedev/logger"
import {
  Array as Arr,
  Deferred,
  Effect,
  Option,
  Schema,
  Scope,
} from "effect"
import type { SessionOptions } from "@magnitudedev/sdk"
import type { CliAppProps, SessionStart } from "../app"
import type { AuthSource } from "../state/cli-atoms"
import { getLastSessionId } from "../state/last-session"
import { CLI_VERSION } from "../version"
import { makeCliEffectLoggingLayer } from "../platform/effect-logger"
import {
  detectTerminalAppearance,
  installTerminalAppearanceRuntime,
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
  onUpdateSelect: (outcome: UpdatePromptOutcome) => void,
): Effect.Effect<void> => Effect.sync(() => {
  root.render(
    <RegistryContext.Provider value={registry}>
      <CliStartupRoot
        stateAtom={stateAtom}
        onUpdateSelect={onUpdateSelect}
      />
    </RegistryContext.Provider>,
  )
})

const raceExit = <A, R,>(
  effect: Effect.Effect<A, never, R>,
  exit: Effect.Effect<ProcessExitRequest>,
): Effect.Effect<ExitRace<A>, never, R> => Effect.race(
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
  CliUpdater | Scope.Scope
> => Effect.gen(function* () {
  const updater = yield* CliUpdater
  const registry = yield* acquireRegistry
  const effectLoggingLayer = makeCliEffectLoggingLayer({
    debug: options.debug,
    publishNotification: (notification) => {
      registry.set(pushNotificationAtom, notification)
    },
  })
  const renderer = yield* acquireRenderer
  const appearance = yield* detectTerminalAppearance(renderer)
  registry.set(terminalAppearanceAtom, appearance)
  yield* installTerminalAppearanceRuntime(renderer, registry)
  const processExit = yield* makeProcessExitSource
  const updateSelection = yield* Deferred.make<UpdatePromptOutcome>()
  const stateAtom = makeCliRootStateAtom()
  const root = yield* acquireRoot(renderer)
  yield* renderRoot(
    root,
    registry,
    stateAtom,
    (outcome) => Deferred.unsafeDone(updateSelection, Effect.succeed(outcome)),
  )

  const shouldPromptForUpdate =
    (options.initialPrompt?.length ?? 0) === 0
    && process.stdin.isTTY === true
    && process.stdout.isTTY === true
  const upgradeVersion = shouldPromptForUpdate
    ? yield* updater.getUpgradeVersion.pipe(Effect.provide(effectLoggingLayer))
    : Option.none()

  if (Option.isSome(upgradeVersion) && Option.isSome(updater.updateAction)) {
    registry.set(stateAtom, {
      _tag: "UpdateAvailable",
      currentVersion: CLI_VERSION,
      latestVersion: upgradeVersion.value,
      action: updater.updateAction.value,
    })
    const selected = yield* raceExit(Deferred.await(updateSelection), processExit.await)
    if (selected._tag === "Exit") return preApplicationExit(selected.request)
    if (selected.value._tag === "Dismiss") {
      yield* updater.dismissVersion(upgradeVersion.value)
    }
    if (selected.value._tag === "Update") {
      return { _tag: "UpdateRequested", action: updater.updateAction.value }
    }
  }

  registry.set(stateAtom, { _tag: "Starting", stage: "Platform" })
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

  registry.set(stateAtom, { _tag: "Starting", stage: "ClientPreflight" })
  const prepared = yield* raceExit(Effect.gen(function* () {
    const initialAcnLifecycle = yield* terminal.platform.acnStartup.prepare
    const agentClient = createAgentClient(terminal.platform.protocolLayer, {
      onboardingSetupInitiallyOpen: options.setup,
    })
    yield* Effect.exit(Registry.getResult(
      registry,
      onboardingModelSetupViewAtom(agentClient),
    ))
    return { initialAcnLifecycle, agentClient }
  }), processExit.await)
  if (prepared._tag === "Exit") return preApplicationExit(prepared.request)

  const app: CliAppProps = {
    sessionStart: options.sessionStart,
    initialPrompt: options.initialPrompt,
    goal: options.goal,
    envAuth: options.envAuth,
    initialAcnLifecycle: prepared.value.initialAcnLifecycle,
    sessionOptions: options.sessionOptions,
  }
  registry.set(stateAtom, {
    _tag: "Application",
    platform: terminal.platform,
    agentClient: prepared.value.agentClient,
    app,
  })

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
    yield* executeUpdate(updater, result.action)
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
