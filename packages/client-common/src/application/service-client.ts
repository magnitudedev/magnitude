import type { HttpClient } from "@effect/platform"
import { Effect, Layer, Option, Stream } from "effect"
import { MagnitudeClient, MagnitudeServiceStarter, ServiceStartFailed } from "@magnitudedev/sdk"
import type { OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import type { ApplicationHost, HostShell } from "./host"

type Settled = Exclude<OwnedServiceState, { readonly _tag: "Starting" }>

const settled = (service: Settled): Effect.Effect<void, ServiceStartFailed> => {
  switch (service._tag) {
    case "Ready": return Effect.void
    case "Failed":
    case "CleanupFailed": return Effect.fail(new ServiceStartFailed({ message: service.message }))
    case "Stopping":
    case "Stopped": return Effect.fail(new ServiceStartFailed({ message: "Magnitude stopped its service." }))
  }
}

/**
 * A desktop window's owner starts the service, so the window's service starts when the owner says it is ready, and
 * fails with the owner's own message when it could not start. Until then the service may not be listening at all,
 * which is starting, not unavailable.
 */
const ownerStarter = (application: HostShell["application"]): MagnitudeServiceStarter => ({
  start: application.pipe(
    Stream.map((snapshot) => snapshot.service),
    Stream.filter((service): service is Settled => service._tag !== "Starting"),
    Stream.runHead,
    Effect.mapError((error) => new ServiceStartFailed({ message: error.message })),
    Effect.flatMap(Option.match({
      onNone: () => Effect.fail(new ServiceStartFailed({ message: "Magnitude stopped reporting its service." })),
      onSome: settled,
    })),
    Stream.fromEffect,
    Stream.drain,
  ),
})

/**
 * The SDK client for one application window. A desktop window waits on its owner while the service starts; a browser
 * is served by the service itself, so one it cannot reach is unavailable.
 */
export const applicationClientLayer = (options: {
  readonly origin: string
  readonly host: ApplicationHost
  readonly reconnect: "Automatic" | "OnConnect"
}): Layer.Layer<MagnitudeClient, never, HttpClient.HttpClient> =>
  Option.match(options.host.shell, {
    onNone: () => MagnitudeClient.layer({ origin: options.origin, autoStart: false, reconnect: options.reconnect }),
    onSome: (shell) => MagnitudeClient.layer({ origin: options.origin, reconnect: options.reconnect }).pipe(
      Layer.provide(Layer.succeed(MagnitudeServiceStarter, ownerStarter(shell.application)))),
  })
