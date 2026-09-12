import { Effect, Option } from "effect"
import type { HarnessConnectionPaths } from "../paths"
import {
  ANTHROPIC_BASE_URL,
  anthropicLocalModelId,
  defineConnector,
  isAnthropicLocalModelId,
  jsonObject,
  readOr,
  removeOwnedJsonc,
  updateJsonc,
  valueAt,
  writeIfChanged,
} from "../shared"

export const CLAUDE_GATEWAY_DISCOVERY = "1"

export const makeClaudeCodeConnector = (paths: HarnessConnectionPaths, baseUrl = ANTHROPIC_BASE_URL) => defineConnector({
  id: "claude-code",
  name: "Claude Code",
  executable: "claude",
  skillInstallationTarget: "claude-user",
  configurationFiles: [paths.claude],
  connect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.claude, "{}\n")
    const value = jsonObject(source)
    const previous = valueAt(value, ["model"])
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [
      [["env", "ANTHROPIC_BASE_URL"], baseUrl],
      [["env", "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], CLAUDE_GATEWAY_DISCOVERY],
    ]
    if (Option.isSome(spec.model)) changes.push([["model"], anthropicLocalModelId(spec.model.value)])
    yield* writeIfChanged(paths.claude, source, updateJsonc(source, changes))
    return Option.map(spec.model, () => ({
      model: typeof previous === "string" ? Option.some(previous) : Option.none(),
    }))
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.claude, "{}\n")
    const value = jsonObject(source)
    const current = valueAt(value, ["model"])
    const currentBaseUrl = valueAt(value, ["env", "ANTHROPIC_BASE_URL"])
    const withoutGateway = removeOwnedJsonc(source, [
      [["env", "ANTHROPIC_BASE_URL"], baseUrl],
      [["env", "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], CLAUDE_GATEWAY_DISCOVERY],
    ])
    const next = isAnthropicLocalModelId(current) && currentBaseUrl === baseUrl && Option.isSome(spec.restore)
      ? updateJsonc(withoutGateway, [[["model"], Option.getOrUndefined(spec.restore.value.model)]])
      : withoutGateway
    yield* writeIfChanged(paths.claude, source, next)
  }),
})
