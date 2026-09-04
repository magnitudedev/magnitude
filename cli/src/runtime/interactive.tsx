import { resolve } from "path"
import type { Root } from "@opentui/react"
import type { CliRenderer } from "@opentui/core"
import { createCliRenderer } from "@opentui/core"
import { createRoot } from "@opentui/react"
import {
  Registry,
  RegistryContext,
  Result,
  scheduleTask,
} from "@effect-atom/atom-react"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import type * as HttpClient from "@effect/platform/HttpClient"
import type * as Path from "@effect/platform/Path"
import type * as Terminal from "@effect/platform/Terminal"
import { FetchHttpClient } from "@effect/platform"
import {
  createAgentClient,
  onboardingModelSetupViewAtom,
  pushNotificationAtom,
  stopDisplayViewController,
  type HarnessLaunchPlan,
  type HarnessConnectionError,
} from "@magnitudedev/client-common"
import {
  updateActionFor,
  type PackageManager,
  type UpdateAction,
} from "@magnitudedev/release"
import { logger } from "@magnitudedev/logger"
import {
  acnInstallationPresent,
  SDK_ACN_TARGET,
  type AcnConnection,
  type AcnEnsuranceError,
} from "@magnitudedev/sdk"
import {
  interactiveProcessExitCode,
  runInteractiveProcess,
} from "@magnitudedev/utils/process"
import type { StateDocumentError } from "@magnitudedev/storage"
import { Array as Arr, Effect, Fiber, Layer, Option, Schema, Scope, Stream } from "effect"
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
  makeTerminalAdapter,
} from "../platform/terminal"
import {
  makeProcessExitSource,
  restoreTerminalState,
  type ProcessExitRequest,
} from "../platform/process-exit"
import { terminalAppearanceAtom } from "../hooks/use-theme"
import { executeUpdate } from "../features/update/execute"
import { CliUpdater, makeCliUpdater } from "../features/update/updater"
import { CliApplicationRoot } from "./root"
import { makeHarnessConnection } from "../harness-connections/service"
import { makeAcnConnectionWithInstanceManager } from "../server/acn-connection"
import { makeBootstrappingAcnInstanceManager } from "../server/acn-instance-manager"
import { resolveCliTheme } from "../utils/theme"
import { runInlineUpdatePrompt } from "../startup/inline-update-prompt"
import { makeInlineServiceStartupPresenter } from "../startup/inline-service-lifecycle"

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
  | {
      readonly _tag: "LaunchHarness"
      readonly plan: HarnessLaunchPlan
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
  platform: Parameters<typeof CliApplicationRoot>[0]["platform"],
  agentClient: Parameters<typeof CliApplicationRoot>[0]["agentClient"],
  startup: Parameters<typeof CliApplicationRoot>[0]["startup"],
  app: CliAppProps,
): Effect.Effect<void> => Effect.sync(() => {
  root.render(
    <RegistryContext.Provider value={registry}>
      <CliApplicationRoot
        platform={platform}
        agentClient={agentClient}
        startup={startup}
        app={app}
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
  connection: AcnConnection,
  request: ProcessExitRequest,
): Effect.Effect<InteractiveSessionResult> => request._tag === "Fatal"
  ? Effect.succeed(fatalResult(request))
  : connection.close.pipe(
      Effect.map(() => {
        const notices: string[] = []
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
  CliRendererAcquisitionFailed | StateDocumentError | AcnEnsuranceError | HarnessConnectionError,
  CliUpdater | FileSystem.FileSystem | Path.Path | Scope.Scope
    | CommandExecutor.CommandExecutor | HttpClient.HttpClient | Terminal.Terminal
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

  // The appearance probe starts with the session — concurrent with discovery
  // and service work — and self-terminates within one terminal roundtrip
  // (fence) or its 100ms ceiling, so its answer is in long before anything
  // paints. It owns terminal input until then; the renderer is only created
  // after it is joined.
  const appearanceProbe = yield* Effect.forkScoped(probeTerminalAppearance())

  // Update interaction happens only on plain interactive launches with a
  // known owning package manager; discovery itself still runs and caches.
  const updateMethod: Option.Option<PackageManager> =
    (options.initialPrompt?.length ?? 0) === 0
    && process.stdin.isTTY === true
    && process.stdout.isTTY === true
      ? updater.packageManager
      : Option.none()
  const discovery = yield* updater.discover.pipe(Effect.provide(effectLoggingLayer))
  let freshPending = true
  const declinedVersions = new Set<string>()
  const offerable = (latest: Option.Option<string>): Option.Option<string> =>
    Option.filter(latest, (version) => !declinedVersions.has(version))

  const presentUpdatePrompt = (latestVersion: string, action: UpdateAction) =>
    Effect.gen(function* () {
      const appearance = yield* Fiber.join(appearanceProbe)
      const selected = yield* raceExit(runInlineUpdatePrompt({
        currentVersion: CLI_VERSION,
        latestVersion,
        action,
        theme: resolveCliTheme(appearance),
      }), processExit.await)
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

  if (Option.isSome(updateMethod) && Option.isSome(discovery.known)) {
    const action = updateActionFor(updateMethod.value, discovery.known.value)
    const resolution = yield* presentUpdatePrompt(discovery.known.value, action)
    if (resolution._tag === "Exit") return preApplicationExit(resolution.request)
    if (resolution._tag === "Update") {
      return { _tag: "UpdateRequested", action }
    }
  }

  // The installation gate: with no installed service build this launch is about
  // to download one, so it awaits the version check first — an offer must
  // prompt before any download of the version it would replace. Installation
  // needs the network regardless, so the wait costs nothing real.
  if (Option.isSome(updateMethod)
    && !(yield* acnInstallationPresent(SDK_ACN_TARGET.identity))) {
    const answer = yield* raceExit(discovery.fresh, processExit.await)
    if (answer._tag === "Exit") return preApplicationExit(answer.request)
    freshPending = false
    const offer = offerable(answer.value)
    if (Option.isSome(offer)) {
      const action = updateActionFor(updateMethod.value, offer.value)
      const resolution = yield* presentUpdatePrompt(offer.value, action)
      if (resolution._tag === "Exit") return preApplicationExit(resolution.request)
      if (resolution._tag === "Update") {
        return { _tag: "UpdateRequested", action }
      }
    }
  }

  const connectionResult = yield* raceExit(
    Effect.gen(function* () {
      const manager = yield* makeBootstrappingAcnInstanceManager({
        launchCommand: developmentLaunchCommand(options),
        debug: options.debug,
      })
      return yield* makeAcnConnectionWithInstanceManager(manager)
    }),
    processExit.await,
  )
  if (connectionResult._tag === "Exit") return preApplicationExit(connectionResult.request)
  const connection = connectionResult.value
  const appearance = yield* Fiber.join(appearanceProbe)
  const startupTheme = resolveCliTheme(appearance)
  const serviceStartup = yield* makeInlineServiceStartupPresenter(startupTheme)

  type ServiceWaitOutcome =
    | { readonly _tag: "Ready" }
    | { readonly _tag: "FreshOffer"; readonly version: string }
    | { readonly _tag: "Exit"; readonly request: ProcessExitRequest }

  // A fresh update offer may interrupt terminal presentation, but it does not
  // interrupt the shared SDK ensurance occurrence. Declining resumes from the
  // current authoritative lifecycle state.
  let starting = true
  while (starting) {
    const readiness = Effect.race(
      serviceStartup.run(connection.startup).pipe(
        Effect.map((): ServiceWaitOutcome => ({ _tag: "Ready" })),
      ),
      processExit.await.pipe(Effect.map(
        (request): ServiceWaitOutcome => ({ _tag: "Exit", request }),
      )),
    )
    const outcome = Option.isSome(updateMethod) && freshPending
      ? yield* Effect.race(readiness, discovery.fresh.pipe(
        Effect.flatMap((latest) => Option.match(offerable(latest), {
          onNone: () => Effect.never,
          onSome: (offer): Effect.Effect<ServiceWaitOutcome> => Effect.succeed({
            _tag: "FreshOffer",
            version: offer,
          }),
        })),
      ))
      : yield* readiness
    if (outcome._tag === "Exit") return preApplicationExit(outcome.request)
    if (outcome._tag === "Ready") {
      starting = false
      continue
    }
    freshPending = false
    if (Option.isNone(updateMethod)) continue
    const action = updateActionFor(updateMethod.value, outcome.version)
    const resolution = yield* presentUpdatePrompt(outcome.version, action)
    if (resolution._tag === "Exit") return preApplicationExit(resolution.request)
    if (resolution._tag === "Update") {
      return { _tag: "UpdateRequested", action }
    }
  }

  const connected = yield* raceExit(Effect.gen(function* () {
    const harnessConnection = yield* makeHarnessConnection
    const protocolLayer = connection.protocolLayer.pipe(Layer.provide(
      Layer.mergeAll(FetchHttpClient.layer, effectLoggingLayer),
    ))
    const agentClient = createAgentClient(protocolLayer, {
      onboardingSetupInitiallyOpen: options.setup,
      harnessConnection,
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
    envAuth: options.envAuth,
    sessionOptions: options.sessionOptions,
  }
  registry.set(terminalAppearanceAtom, appearance)
  const renderer = yield* acquireRenderer
  yield* installTerminalAppearanceRuntime(renderer, registry)
  const root = yield* acquireRoot(renderer)
  yield* renderRoot(
    root,
    registry,
    makeTerminalAdapter(),
    connected.value,
    connection.startup,
    app,
  )

  // A discovery answer arriving after the app committed surfaces as one
  // notification line; the prompt would interrupt real work now.
  if (Option.isSome(updateMethod) && freshPending) {
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

  const handoff = Registry.toStream(
    registry,
    onboardingModelSetupViewAtom(connected.value),
  ).pipe(
    Stream.filterMap((result) => Option.flatMap(Result.value(result), (state) =>
      state._tag === "Open" && state.content._tag === "HarnessHandoff"
        ? Option.some(state.content.plan)
        : Option.none())),
    Stream.runHead,
    Effect.flatMap(Option.match({ onNone: () => Effect.never, onSome: Effect.succeed })),
  )
  const outcome = yield* Effect.race(
    processExit.await.pipe(Effect.map((request) => ({ _tag: "Exit" as const, request }))),
    handoff.pipe(Effect.map((plan) => ({ _tag: "Launch" as const, plan }))),
  )
  if (outcome._tag === "Exit") return yield* closeApplication(connection, outcome.request)
  yield* connection.close.pipe(Effect.ignore)
  stopDisplayViewController()
  return { _tag: "LaunchHarness", plan: outcome.plan }
})

const writeSessionResult = (
  result: Extract<InteractiveSessionResult, { readonly _tag: "Exit" }>,
): Effect.Effect<void> => Effect.sync(() => {
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
): Effect.Effect<
  number,
  CliRendererAcquisitionFailed | StateDocumentError | AcnEnsuranceError | HarnessConnectionError,
  | CommandExecutor.CommandExecutor
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | Path.Path
  | Terminal.Terminal
> => Effect.gen(function* () {
  const updater = yield* makeCliUpdater({
    currentVersion: CLI_VERSION,
    developmentBuild: options.developmentBuild,
  })
  const result = yield* Effect.scoped(runInteractiveSession(options).pipe(
    Effect.provideService(CliUpdater, updater),
  ))
  if (result._tag === "UpdateRequested") {
    return yield* executeUpdate(updater, result.action, { relaunch: true })
  }
  if (result._tag === "LaunchHarness") {
    return yield* runInteractiveProcess({
      executable: result.plan.executable,
      args: result.plan.args,
      environment: { ...process.env, ...result.plan.environment },
      workingDirectory: process.cwd(),
    }).pipe(
      Effect.map(interactiveProcessExitCode),
      Effect.catchAll((error) => Effect.sync(() => {
        const manual = [result.plan.command, ...result.plan.args]
          .map((part) => /[\s'"\\]/.test(part) ? JSON.stringify(part) : part)
          .join(" ")
        process.stderr.write([
          `Magnitude setup is complete, but ${result.plan.harness} could not be launched.`,
          String(error),
          `Run manually: ${manual}`,
          "",
        ].join("\n"))
        return 1
      })),
    )
  }
  yield* writeSessionResult(result)
  return result.code
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
