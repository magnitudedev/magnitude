import * as HttpClient from "@effect/platform/HttpClient"
import { RpcClient, RpcClientError } from "@effect/rpc"
import { Effect, Fiber, Layer, Option, Stream } from "effect"
import {
  isInterruptedExit,
  makeJitDaemonCoordinator,
  recoveringProtocolLayer as jitRecoveringProtocolLayer,
} from "../jit-rpc"
import type { JitDaemonCoordinator } from "../jit-rpc"
import {
  DaemonSpawnerTag,
  runDaemonSpawn,
} from "./daemon-spawner"
import type { DaemonError } from "./errors"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"
import {
  makeAcnLifecycle,
  type AcnLifecycle,
  type AcnLifecycleState,
} from "./lifecycle"
import { SDK_VERSION } from "../version"

export interface AcnJitRuntimeOptions {
  /** Explicit ACN spawn command; when omitted the spawner resolves the binary. */
  readonly spawnCommand: Option.Option<ReadonlyArray<string>>
}

export interface AcnStartup {
  readonly state: AcnLifecycle
  /**
   * Performs pre-render discovery. When no daemon is ready, starts the shared
   * ensure attempt and returns its first user-visible lifecycle state.
   */
  readonly prepare: Effect.Effect<AcnLifecycleState>
  /** Retries the shared ensure operation after a startup failure. */
  readonly retry: Effect.Effect<void, DaemonError>
}

/** One process-local ACN runtime shared by every RPC consumer. */
export interface AcnJitRuntime {
  readonly protocolLayer: Layer.Layer<
    RpcClient.Protocol,
    never,
    HttpClient.HttpClient
  >
  readonly startup: AcnStartup
}

const { RpcClientError: TransportError } = RpcClientError

const unavailableError = (cause: DaemonError): RpcClientError.RpcClientError =>
  new TransportError({
    reason: "Unknown",
    message: `ACN unavailable: ${cause._tag}`,
    cause,
  })

/**
 * The sole ACN JIT composition entrypoint. It creates one coordinator and
 * returns one protocol layer for every RPC consumer. Construction is
 * non-blocking; explicit ensure and ordinary RPC recovery share that
 * coordinator. Subscription framing remains inside the ACN adapter.
 */
export const makeAcnJitRuntime = (
  options: AcnJitRuntimeOptions = { spawnCommand: Option.none() }
): Effect.Effect<AcnJitRuntime, never, DaemonSpawnerTag> =>
  Effect.gen(function* () {
    const spawner = yield* DaemonSpawnerTag
    const lifecycle = yield* makeAcnLifecycle()
    const provider = {
      discover: () =>
        spawner.discover().pipe(
          Effect.map(Option.map((url) => ({ url }))),
        ),
      spawn: () =>
        runDaemonSpawn(
          spawner.spawn(options.spawnCommand).pipe(
            Stream.tap((event) =>
              event._tag === "Observation"
                ? lifecycle.report(event.observation)
                : Effect.void,
            ),
          ),
        ).pipe(Effect.map((url) => ({ url }))),
    }
    const coordinator = yield* makeJitDaemonCoordinator(provider)

    const ready = (url: string) => lifecycle.ready(url, SDK_VERSION)

    const discover = coordinator.discover.pipe(
      Effect.tap(
        Option.match({
          onNone: () => Effect.void,
          onSome: (lease) => ready(lease.endpoint.url),
        })
      ),
      Effect.map(Option.map((lease) => lease.endpoint.url)),
      Effect.tapError(lifecycle.fail)
    )

    const ensure = coordinator.ensure.pipe(
      Effect.tap((lease) => ready(lease.endpoint.url)),
      Effect.asVoid,
      Effect.tapError(lifecycle.fail)
    )
    const prepare = Effect.gen(function* () {
      yield* discover.pipe(Effect.ignore)
      const discovered = yield* lifecycle.get
      if (discovered._tag !== "Checking") return discovered

      const firstVisible = yield* lifecycle.changes.pipe(
        Stream.filter((state) => state._tag !== "Checking"),
        Stream.runHead,
        Effect.fork,
      )
      yield* ensure.pipe(Effect.ignore, Effect.forkDaemon)
      const observed = yield* Fiber.join(firstVisible)
      return yield* Option.match(observed, {
        onNone: () =>
          Effect.dieMessage(
            "ACN lifecycle ended before publishing a visible startup state",
          ),
        onSome: Effect.succeed,
      })
    })

    const recoveringCoordinator: JitDaemonCoordinator<DaemonError> = {
      ...coordinator,
      ensure: coordinator.ensure.pipe(
        Effect.tap((lease) => ready(lease.endpoint.url)),
        Effect.tapError(lifecycle.fail)
      ),
      invalidate: (lease, invalidateOptions) =>
        coordinator.invalidate(lease, invalidateOptions).pipe(
          Effect.zipRight(coordinator.current),
          Effect.flatMap(
            Option.match({
              onNone: () =>
                lifecycle.report({ _tag: "Starting", phase: "Discovering" }),
              onSome: () => Effect.void,
            })
          )
        ),
      awaitSuccessor: (lease) =>
        lifecycle.report({ _tag: "Starting", phase: "WaitingForOwner" }).pipe(
          Effect.zipRight(coordinator.awaitSuccessor(lease)),
          Effect.tap((successor) => ready(successor.endpoint.url)),
          Effect.tapError(lifecycle.fail)
        ),
    }

    return {
      startup: {
        state: lifecycle,
        prepare,
        retry: ensure,
      },
      protocolLayer: jitRecoveringProtocolLayer({
        coordinator: recoveringCoordinator,
        rpcPath: "/rpc",
        streamProtocol: acnSubscriptionProtocol,
        isEndpointRetirementExit: isInterruptedExit,
        classifyInfraError: unavailableError,
      }),
    }
  })
