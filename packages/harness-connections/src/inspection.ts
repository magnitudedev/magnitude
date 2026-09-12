import { Effect, Option, Schema } from "effect"
import { parse as parseToml } from "smol-toml"
import { parseDocument } from "yaml"
import { isDeepStrictEqual } from "node:util"
import type { HarnessConnector, HarnessModel } from "./contract"
import type { HarnessConnectionPaths } from "./paths"
import { harnessEndpoints, jsonObject, readOr, valueAt } from "./shared"
import { CLAUDE_GATEWAY_DISCOVERY } from "./connectors/claude-code"
import { clineModelCatalog, clineProviderSettings } from "./connectors/cline"
import { hermesProviderConfig } from "./connectors/hermes"
import { ohMyPiProviderConfig } from "./connectors/oh-my-pi"
import { openClawProviderConfig } from "./connectors/openclaw"
import { openCodeProviderConfig } from "./connectors/opencode"
import { piProviderConfig } from "./connectors/pi"

import { ConnectionInspection } from "@magnitudedev/client-common"
export { ConnectionInspection } from "@magnitudedev/client-common"

const disconnected = (reason: string): ConnectionInspection => ({ _tag: "Disconnected", reason })
const connected: ConnectionInspection = { _tag: "Connected" }
const yamlObject = (source: string): unknown => {
  const document = parseDocument(source)
  if (document.errors.length > 0) throw document.errors[0]
  return document.toJS()
}
const includesFields = (actual: unknown, expected: unknown): boolean => {
  if (expected === null || typeof expected !== "object") return isDeepStrictEqual(actual, expected)
  if (Array.isArray(expected)) return Array.isArray(actual) && expected.every((value) => actual.some((candidate) => includesFields(candidate, value)))
  if (actual === null || typeof actual !== "object" || Array.isArray(actual)) return false
  return Object.entries(expected).every(([key, value]) => includesFields(valueAt(actual, [key]), value))
}

/** Read owned configuration fields; never repair files or start a harness while inspecting. */
export const inspectHarnessConnection = (
  paths: HarnessConnectionPaths,
  connector: HarnessConnector,
  models: ReadonlyArray<HarnessModel>,
  requiredSkill: Option.Option<string>,
  serviceEndpoint?: string,
) => Effect.gen(function* () {
  const endpoints = harnessEndpoints(serviceEndpoint)
  const read = (path: string, format: "json" | "yaml" | "toml" = "json") => readOr(path, format === "toml" ? "" : "{}").pipe(
    Effect.flatMap((source) => Effect.try({
      try: () => format === "yaml" ? yamlObject(source) : format === "toml" ? parseToml(source) : jsonObject(source),
      catch: () => disconnected(`Configuration is invalid: ${path}`),
    })),
  )
  const provider = (path: string, segments: ReadonlyArray<string>, expected: unknown, format: "json" | "yaml" = "json") =>
    read(path, format).pipe(Effect.map((document) => includesFields(valueAt(document, segments), expected)))
  let valid = false
  switch (connector.id) {
    case "pi": valid = yield* provider(paths.piModels, ["providers", "magnitude"], piProviderConfig(models, endpoints.openai)); break
    case "opencode": valid = yield* provider(paths.opencode, ["provider", "magnitude"], openCodeProviderConfig(models, endpoints.openai)); break
    case "hermes": valid = yield* provider(paths.hermes, ["providers", "magnitude"], hermesProviderConfig(endpoints.openai), "yaml"); break
    case "openclaw": valid = yield* provider(paths.openclaw, ["models", "providers", "magnitude"], openClawProviderConfig(models, endpoints.openai)); break
    case "oh-my-pi": valid = yield* provider(paths.ompModels, ["providers", "magnitude"], ohMyPiProviderConfig(models, endpoints.openai), "yaml"); break
    case "claude-code": valid = yield* provider(paths.claude, ["env"], {
      ANTHROPIC_BASE_URL: endpoints.anthropic, CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY: CLAUDE_GATEWAY_DISCOVERY,
    }); break
    case "cline":
      valid = (yield* provider(paths.clineProviders, ["providers", "openai-compatible", "settings"], clineProviderSettings(Option.none(), endpoints.openai)))
        && (yield* provider(paths.clineModels, ["providers", "openai-compatible", "models"], clineModelCatalog(models)))
      break
    case "codex": {
      const document = yield* read(paths.codexUser, "toml")
      valid = includesFields(valueAt(document, ["model_providers", "magnitude"]), {
        base_url: endpoints.codex, wire_api: "responses", requires_openai_auth: true, supports_websockets: true,
      }) && valueAt(document, ["model_catalog_json"]) === paths.codexModels
      if (valid) {
        const catalog = yield* read(paths.codexModels)
        const entries = valueAt(catalog, ["models"])
        valid = Array.isArray(entries) && models.every((model) => entries.some((entry) => valueAt(entry, ["slug"]) === `magnitude-local/${model.id}`))
      }
      break
    }
  }
  if (!valid) return disconnected("Magnitude configuration is missing or has changed")
  if (Option.isSome(requiredSkill)) {
    const path = paths.skillInstallations[connector.skillInstallationTarget].skillFile
    if ((yield* readOr(path, "")) !== requiredSkill.value) return disconnected("Magnitude skill is missing or has changed")
  }
  if (connector.companion !== undefined && !(yield* connector.companion.inspect)) {
    return disconnected("Magnitude plugin is missing, disabled, or incompatible")
  }
  return connected
}).pipe(
  Effect.catchAll((error) => Effect.succeed<ConnectionInspection>(
    Schema.is(ConnectionInspection)(error) ? error
      : { _tag: "Unavailable", reason: String(error) },
  )),
)
