import { Effect } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import { defineConnector, launchPlan } from "../shared"

export const makeMagnitudeConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "magnitude",
  name: "Magnitude",
  executable: "magnitude",
  recommended: true,
  note: "Optimized for local models",
  skillFile: paths.skills.magnitude!,
  configurationFiles: [],
  connect: () => Effect.void,
  disconnect: () => Effect.void,
  launch(modelId, installation) { return launchPlan(this, installation, modelId, []) },
})
