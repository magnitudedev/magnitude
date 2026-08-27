import { Effect } from "effect"
import { defineConnector, launchPlan } from "../shared"

export const makeMagnitudeConnector = () => defineConnector({
  id: "magnitude",
  name: "Magnitude Harness",
  executable: "magnitude",
  recommended: true,
  note: "Optimized for local models",
  skillInstallationTarget: "shared-agents",
  configurationFiles: [],
  connect: () => Effect.void,
  disconnect: () => Effect.void,
  launch(modelId, installation) { return launchPlan(this, installation, modelId, []) },
})
