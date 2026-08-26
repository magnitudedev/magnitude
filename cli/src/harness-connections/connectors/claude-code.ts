import { Effect, Option } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  ANTHROPIC_BASE_URL,
  LOCAL_TOKEN,
  anthropicLocalModelId,
  defineConnector,
  launchPlan,
  readOr,
  removeOwnedJsonc,
  writeIfChanged,
} from "../shared"

export const makeClaudeCodeConnector = (paths: HarnessConnectionPaths) => defineConnector({
  id: "claude-code",
  name: "Claude Code",
  executable: "claude",
  skillFile: paths.skills["claude-code"]!,
  configurationFiles: [paths.claude],
  connect: () => Effect.void,
  disconnect: (spec) => readOr(paths.claude, "{}\n").pipe(
    Effect.flatMap((source) => {
      const owned = Option.match(spec.setCurrent, {
        onNone: () => [],
        onSome: (modelId) => {
          const localModelId = anthropicLocalModelId(modelId)
          return [
            [["env", "ANTHROPIC_BASE_URL"], ANTHROPIC_BASE_URL] as const,
            [["env", "ANTHROPIC_AUTH_TOKEN"], LOCAL_TOKEN] as const,
            [["env", "ANTHROPIC_MODEL"], localModelId] as const,
            [["env", "ANTHROPIC_DEFAULT_HAIKU_MODEL"], localModelId] as const,
            [["env", "ANTHROPIC_DEFAULT_SONNET_MODEL"], localModelId] as const,
            [["env", "ANTHROPIC_DEFAULT_OPUS_MODEL"], localModelId] as const,
          ]
        },
      })
      return writeIfChanged(paths.claude, source, removeOwnedJsonc(source, owned))
    }),
  ),
  launch(modelId, installation) {
    const localModelId = anthropicLocalModelId(modelId)
    return launchPlan(this, installation, modelId, ["--model", localModelId], {
      ANTHROPIC_BASE_URL,
      ANTHROPIC_AUTH_TOKEN: LOCAL_TOKEN,
      ANTHROPIC_MODEL: localModelId,
      ANTHROPIC_DEFAULT_HAIKU_MODEL: localModelId,
      ANTHROPIC_DEFAULT_SONNET_MODEL: localModelId,
      ANTHROPIC_DEFAULT_OPUS_MODEL: localModelId,
    })
  },
})
