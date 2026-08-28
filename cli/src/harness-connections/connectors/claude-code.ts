import { Effect } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  ANTHROPIC_BASE_URL,
  defineConnector,
  launchPlan,
  readOr,
  removeOwnedJsonc,
  updateJsonc,
  writeIfChanged,
} from "../shared"

export const CLAUDE_GATEWAY_DISCOVERY = "1"

export const makeClaudeCodeConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "claude-code",
  name: "Claude Code",
  executable: "claude",
  requiresStartup: true,
  skillInstallationTarget: "claude-user",
  configurationFiles: [paths.claude],
  connect: () => readOr(paths.claude, "{}\n").pipe(Effect.flatMap((source) =>
    writeIfChanged(paths.claude, source, updateJsonc(source, [
      [["env", "ANTHROPIC_BASE_URL"], ANTHROPIC_BASE_URL],
      [["env", "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], CLAUDE_GATEWAY_DISCOVERY],
    ])))),
  disconnect: () => readOr(paths.claude, "{}\n").pipe(Effect.flatMap((source) =>
    writeIfChanged(paths.claude, source, removeOwnedJsonc(source, [
      [["env", "ANTHROPIC_BASE_URL"], ANTHROPIC_BASE_URL],
      [["env", "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], CLAUDE_GATEWAY_DISCOVERY],
    ])))),
  launch(modelId, installation) {
    return launchPlan(this, installation, modelId, ["--model", `anthropic-local/${modelId}`])
  },
})
