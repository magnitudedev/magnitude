import { Command } from "@effect/platform"
import { Effect, Option } from "effect"
import { isDeepStrictEqual } from "node:util"
import { MAGNITUDE_CLAUDE_CODE_PLACEHOLDER_CREDENTIAL } from "@magnitudedev/sdk"
import type { HarnessConnectionPaths } from "../paths"
import {
  ANTHROPIC_BASE_URL,
  anthropicLocalModelId,
  defineConnector,
  isAnthropicLocalModelId,
  jsonObject,
  readOr,
  removeJsoncPaths,
  removeOwnedJsonc,
  updateJsonc,
  valueAt,
  writeIfChanged,
} from "../shared"

export const CLAUDE_GATEWAY_DISCOVERY = "1"
/**
 * Claude Code refuses to send any request without a credential. Without a login, the connection
 * writes Magnitude's placeholder bearer (no approval prompt, unlike `ANTHROPIC_API_KEY`) and an
 * empty `ANTHROPIC_API_KEY` so an exported key cannot take over. The gateway
 * serves every placeholder request locally. A placeholder outranks a Claude login, so it is written
 * only while Claude Code has no credential of its own.
 */
export const CLAUDE_AUTH_TOKEN_PLACEHOLDER = MAGNITUDE_CLAUDE_CODE_PLACEHOLDER_CREDENTIAL

const nonEmptyString = (value: unknown): boolean => typeof value === "string" && value !== ""

const fileHasField = (file: string, segments: ReadonlyArray<string>) => readOr(file, "").pipe(
  Effect.flatMap((source) => Effect.try(() => source !== "" && nonEmptyString(valueAt(jsonObject(source), segments)))),
  Effect.orElseSucceed(() => false),
)

const keychainItemExists = (service: string, account: string) =>
  Command.make("security", "find-generic-password", "-a", account, "-s", service).pipe(
    Command.stdout("pipe"),
    Command.stderr("pipe"),
    Command.exitCode,
    Effect.timeout("5 seconds"),
    Effect.map((code) => code === 0),
    Effect.orElseSucceed(() => false),
  )

/**
 * Whether Claude Code has a credential of its own, checked the way Claude Code finds one. Only the
 * existence of each credential is read; the Keychain lookup never requests the secret.
 */
export const claudeCodeHasLogin = (paths: HarnessConnectionPaths, settings: unknown) => Effect.gen(function* () {
  const env = valueAt(settings, ["env"])
  const token = valueAt(env, ["ANTHROPIC_AUTH_TOKEN"])
  if (nonEmptyString(valueAt(env, ["ANTHROPIC_API_KEY"])) || nonEmptyString(valueAt(env, ["CLAUDE_CODE_OAUTH_TOKEN"]))
      || (nonEmptyString(token) && token !== CLAUDE_AUTH_TOKEN_PLACEHOLDER)
      || nonEmptyString(valueAt(settings, ["apiKeyHelper"]))) return true
  for (const file of paths.claudeLogin.credentialFiles) {
    if (yield* fileHasField(file, ["claudeAiOauth", "accessToken"])) return true
  }
  for (const file of paths.claudeLogin.globalConfigFiles) {
    if (yield* fileHasField(file, ["primaryApiKey"])) return true
  }
  for (const { service, account } of paths.claudeLogin.keychainItems) {
    if (yield* keychainItemExists(service, account)) return true
  }
  return false
})

/** Magnitude's no-login credential values, removed once a login exists and on disconnect. */
const noLoginEnv: ReadonlyArray<readonly [ReadonlyArray<string>, string]> = [
  [["env", "ANTHROPIC_AUTH_TOKEN"], CLAUDE_AUTH_TOKEN_PLACEHOLDER],
  [["env", "ANTHROPIC_API_KEY"], ""],
]

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
    const loggedIn = yield* claudeCodeHasLogin(paths, value)
    const changes: Array<readonly [ReadonlyArray<string>, unknown]> = [
      [["env", "ANTHROPIC_BASE_URL"], baseUrl],
      [["env", "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], CLAUDE_GATEWAY_DISCOVERY],
    ]
    if (!loggedIn) changes.push(...noLoginEnv)
    if (Option.isSome(spec.model)) changes.push([["model"], anthropicLocalModelId(spec.model.value)])
    const base = loggedIn ? removeOwnedJsonc(source, noLoginEnv) : source
    yield* writeIfChanged(paths.claude, source, updateJsonc(base, changes))
    return Option.map(spec.model, () => ({
      model: typeof previous === "string" ? Option.some(previous) : Option.none(),
    }))
  }),
  disconnect: (spec) => Effect.gen(function* () {
    const source = yield* readOr(paths.claude, "{}\n")
    const value = jsonObject(source)
    const current = valueAt(value, ["model"])
    const currentBaseUrl = valueAt(value, ["env", "ANTHROPIC_BASE_URL"])
    const withoutGatewayEnv = removeOwnedJsonc(source, [
      [["env", "ANTHROPIC_BASE_URL"], baseUrl],
      [["env", "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], CLAUDE_GATEWAY_DISCOVERY],
      ...noLoginEnv,
    ])
    const withoutGateway = isDeepStrictEqual(valueAt(jsonObject(withoutGatewayEnv), ["env"]), {})
      ? removeJsoncPaths(withoutGatewayEnv, [["env"]])
      : withoutGatewayEnv
    const next = isAnthropicLocalModelId(current) && currentBaseUrl === baseUrl && Option.isSome(spec.restore)
      ? updateJsonc(withoutGateway, [[["model"], Option.getOrUndefined(spec.restore.value.model)]])
      : withoutGateway
    yield* writeIfChanged(paths.claude, source, next)
  }),
})
