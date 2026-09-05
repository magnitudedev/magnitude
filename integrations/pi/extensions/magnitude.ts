import type { ExtensionAPI } from "@earendil-works/pi-coding-agent"
import type { Model } from "@earendil-works/pi-ai"
import { openAICompletionsApi } from "@earendil-works/pi-ai/compat"
import { registerMagnitudeCommands } from "./commands"
import { registerMagnitudeOnboarding } from "./onboarding"
import { makeObservingFetch } from "./observing-fetch"
import { makeProgressTracker, type ProgressTracker } from "./progress"
import { Effect, Exit, Scope } from "effect"

export default function magnitudeExtension(pi: ExtensionAPI): void {
  // Pi aliases this public entrypoint in both its bundled and npm extension loaders.
  const completions = openAICompletionsApi()
  let tracker: ProgressTracker | undefined
  let scope: Scope.CloseableScope | undefined
  const disposeCommands = registerMagnitudeCommands(pi)
  const disposeOnboarding = registerMagnitudeOnboarding(pi)
  const perform = (effect: Effect.Effect<void> | undefined) => Effect.runSync((effect ?? Effect.void).pipe(Effect.catchAllCause(() => Effect.void)))

  pi.registerProvider("magnitude", {
    api: "openai-completions",
    streamSimple: (model, context, options) => {
      if (model.api !== "openai-completions") {
        throw new Error(`Magnitude for Pi requires openai-completions, received ${model.api}`)
      }
      const response = tracker && Effect.runSync(tracker.beginResponse(model.name))
      const stream = completions.streamSimple(model as Model<"openai-completions">, context, {
        ...options,
        fetch: scope ? makeObservingFetch(
          options?.fetch ?? globalThis.fetch,
          () => response?.begin ?? Effect.succeed({ observe: () => Effect.void, finish: Effect.void, fail: Effect.void }),
          scope,
        ) : options?.fetch ?? globalThis.fetch,
      })
      if (response && scope) Effect.runFork(Effect.forkIn(
        Effect.tryPromise(() => stream.result()).pipe(
          Effect.flatMap((message) => response.end(message.stopReason !== "error" && message.stopReason !== "aborted")),
          Effect.catchAllCause(() => response.end(false)),
        ), scope,
      ))
      return stream
    },
  })

  pi.on("session_start", async (_event, ctx) => {
    if (scope) await Effect.runPromise(Scope.close(scope, Exit.void))
    scope = Effect.runSync(Scope.make())
    tracker = await Effect.runPromise(makeProgressTracker(ctx.ui).pipe(Scope.extend(scope)))
  })
  pi.on("model_select", (event) => {
    if (event.model.provider !== "magnitude") perform(tracker?.clear)
  })
  pi.on("agent_start", (_event, ctx) => {
    if (ctx.model?.provider === "magnitude") perform(tracker?.startRun(ctx.model.name))
  })
  pi.on("agent_settled", () => perform(tracker?.settleRun))
  pi.on("session_shutdown", async () => {
    await disposeOnboarding()
    if (scope) await Effect.runPromise(Scope.close(scope, Exit.void))
    tracker = undefined
    scope = undefined
    await disposeCommands()
  })
}
