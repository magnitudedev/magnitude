import * as HttpClient from "@effect/platform/HttpClient"
import type { AcnEndpoint } from "@magnitudedev/acn-protocol"
import { RpcClient, RpcClientError } from "@effect/rpc"
import { Effect, Fiber, Layer, Option, Stream } from "effect"
import {
  isInterruptedExit,
  recoveringProtocolLayer as jitRecoveringProtocolLayer,
} from "../jit-rpc"
import { DaemonDiscovery, type DaemonStatus } from "./daemon-discovery"
import { DaemonLauncher, runDaemonLaunch } from "./daemon-launcher"
import type { DaemonError } from "./errors"
import { canUseDaemonVersion } from "./release-precedence"
import { SDK_VERSION } from "../version"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"
import {
  makeAcnLifecycle,
  acnLifecycleObservationFromHealthState,
  type AcnLifecycle,
  type AcnLifecycleState,
} from "./lifecycle"

export interface AcnJitRuntimeOptions {
  /** Explicit ACN launch command; when omitted the launcher resolves it. */
  readonly launchCommand: Option.Option<ReadonlyArray<string>>
}

export interface AcnStartup {
  readonly state: AcnLifecycle
  /**
   * Performs pre-render selection. When no ACN is ready, launches one and
   * returns the first user-visible lifecycle state.
   */
  readonly prepare: Effect.Effect<AcnLifecycleState>
  /** Starts a new connection attempt after a startup failure. */
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
 * The sole client-side owner of ACN connection state and recovery policy.
 * Discovery reports facts, launch performs one mutation, and this lifecycle
 * alone decides which endpoint the client may use or replace.
 */
export const makeAcnJitRuntime = (
  options: AcnJitRuntimeOptions = { launchCommand: Option.none() }
): Effect.Effect<AcnJitRuntime, never, DaemonDiscovery | DaemonLauncher> =>
  Effect.gen(function* () {
    const discovery = yield* DaemonDiscovery
    const launcher = yield* DaemonLauncher
    const lifecycle = yield* makeAcnLifecycle()
    const selectionLock = yield* Effect.makeSemaphore(1)

    const launch = runDaemonLaunch(
      launcher.launch(options.launchCommand).pipe(
        Stream.tap((event) =>
          event._tag === "Observation"
            ? lifecycle.report(event.observation)
            : Effect.void,
        ),
      ),
    )

    const compatible = (status: DaemonStatus): boolean =>
      canUseDaemonVersion(SDK_VERSION, status.version)

    const reportDaemonState = (status: DaemonStatus): Effect.Effect<void> =>
      Option.match(acnLifecycleObservationFromHealthState(status.state), {
        onNone: () => Effect.void,
        onSome: lifecycle.report,
      })

    const awaitUsable = (
      rejectedId: Option.Option<string>,
    ): Effect.Effect<AcnEndpoint, DaemonError> =>
      Effect.suspend(() =>
        discovery.current().pipe(
          Effect.flatMap(
            Option.match({
              onNone: () => Effect.sleep("250 millis").pipe(Effect.zipRight(awaitUsable(rejectedId))),
              onSome: (status) => {
                if (Option.contains(rejectedId, status.id) || !compatible(status)) {
                  return Effect.sleep("250 millis").pipe(Effect.zipRight(awaitUsable(rejectedId)))
                }
                if (status.state._tag === "Ready") return Effect.succeed(status)
                if (status.state._tag === "Starting") {
                  return reportDaemonState(status).pipe(
                    Effect.zipRight(Effect.sleep("250 millis")),
                    Effect.zipRight(awaitUsable(rejectedId)),
                  )
                }
                return Effect.sleep("250 millis").pipe(Effect.zipRight(awaitUsable(rejectedId)))
              },
            }),
          ),
        ),
      )

    const discoverOrLaunch = (
      rejectedId: Option.Option<string>,
      waitMillis: number,
    ): Effect.Effect<AcnEndpoint, DaemonError> =>
      awaitUsable(rejectedId).pipe(
        Effect.timeoutOption(`${waitMillis} millis`),
        Effect.flatMap(Option.match({ onNone: () => launch, onSome: Effect.succeed })),
      )

    const selectEndpoint: Effect.Effect<AcnEndpoint, DaemonError> =
      selectionLock.withPermits(1)(Effect.gen(function* () {
        const state = yield* lifecycle.get
        if (state._tag === "Ready") return state.endpoint

        const current = yield* discovery.current()
        const endpoint = yield* Option.match(current, {
          onNone: () => launch,
          onSome: (status) => {
            if (!compatible(status) || status.state._tag === "Stopping") return launch
            if (status.state._tag === "Ready") return Effect.succeed(status)
            return reportDaemonState(status).pipe(
              Effect.zipRight(discoverOrLaunch(Option.none(), 2_000)),
            )
          },
        })
        yield* lifecycle.ready(endpoint)
        return endpoint
      })).pipe(Effect.tapError(lifecycle.fail))

    const recover = (failed: AcnEndpoint): Effect.Effect<AcnEndpoint, DaemonError> =>
      selectionLock.withPermits(1)(Effect.gen(function* () {
        const state = yield* lifecycle.get
        if (
          state._tag === "Ready" &&
          state.endpoint.id !== failed.id
        ) {
          return state.endpoint
        }

        yield* lifecycle.report({ _tag: "Starting", phase: "Discovering" })
        const endpoint = yield* discoverOrLaunch(Option.some(failed.id), 2_000)
        yield* lifecycle.ready(endpoint)
        return endpoint
      })).pipe(Effect.tapError(lifecycle.fail))

    const prepare = Effect.gen(function* () {
      const initial = yield* lifecycle.get
      if (initial._tag !== "Checking") return initial

      const firstVisible = yield* lifecycle.changes.pipe(
        Stream.filter((state) => state._tag !== "Checking"),
        Stream.runHead,
        Effect.fork,
      )
      yield* selectEndpoint.pipe(Effect.ignore, Effect.forkDaemon)
      const observed = yield* Fiber.join(firstVisible)
      return yield* Option.match(observed, {
        onNone: () =>
          Effect.dieMessage(
            "ACN lifecycle ended before publishing a visible startup state",
          ),
        onSome: Effect.succeed,
      })
    })

    return {
      startup: {
        state: lifecycle,
        prepare,
        retry: selectEndpoint.pipe(Effect.asVoid),
      },
      protocolLayer: jitRecoveringProtocolLayer({
        endpoint: selectEndpoint,
        recover,
        rpcPath: "/rpc",
        streamProtocol: acnSubscriptionProtocol,
        isEndpointRetirementExit: isInterruptedExit,
        classifyInfraError: unavailableError,
      }),
    }
  })
