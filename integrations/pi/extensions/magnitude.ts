import type { ExtensionAPI } from "@earendil-works/pi-coding-agent"
import type { Model } from "@earendil-works/pi-ai"
import { streamSimple } from "@earendil-works/pi-ai/api/openai-completions"
import { registerMagnitudeCommands } from "./commands"
import { makeObservingFetch } from "./observing-fetch"
import { MagnitudeProgressTracker } from "./progress"

export default function magnitudeExtension(pi: ExtensionAPI): void {
  let tracker: MagnitudeProgressTracker | undefined
  const progressTracker = (ui: ConstructorParameters<typeof MagnitudeProgressTracker>[0]) => {
    tracker ??= new MagnitudeProgressTracker(ui)
    return tracker
  }

  pi.registerProvider("magnitude", {
    api: "openai-completions",
    streamSimple: (model, context, options) => {
      if (model.api !== "openai-completions") {
        throw new Error(`Magnitude for Pi requires openai-completions, received ${model.api}`)
      }
      return streamSimple(model as Model<"openai-completions">, context, {
        ...options,
        fetch: makeObservingFetch(
          options?.fetch ?? globalThis.fetch,
          () => tracker?.begin(model.name) ?? {
            observe: () => {},
            finish: () => {},
            fail: () => {},
          },
        ),
      })
    },
  })

  pi.on("session_start", (_event, ctx) => {
    progressTracker(ctx.ui)
  })
  pi.on("model_select", (event) => {
    if (event.model.provider !== "magnitude") tracker?.clear()
  })
  pi.on("agent_start", (_event, ctx) => {
    if (ctx.model?.provider === "magnitude") tracker?.startRun(ctx.model.name)
  })
  pi.on("agent_settled", () => tracker?.settleRun())
  pi.on("session_shutdown", () => tracker?.dispose())
  registerMagnitudeCommands(pi)
}
