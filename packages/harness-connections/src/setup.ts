import { Brand, Option } from "effect"
import { stringify as stringifyToml } from "smol-toml"
import { stringify as stringifyYaml } from "yaml"
import { harnessCommand, harnessInstallationUrl, type HarnessId } from "@magnitudedev/sdk"
import type { HarnessModel } from "./contract"
import { clineModelRegistryEntry, clineProviderSettings } from "./connectors/cline"
import { hermesProviderConfig, hermesReasoningOverrides } from "./connectors/hermes"
import { ohMyPiModels } from "./connectors/oh-my-pi"
import { openClawAgentConfig, openClawModels } from "./connectors/openclaw"
import { openCodeProviderConfig } from "./connectors/opencode"
import { piProviderConfig } from "./connectors/pi"
import { CLAUDE_GATEWAY_DISCOVERY } from "./connectors/claude-code"
import { modelMaxTokens } from "./model-fields"
import { CHAT_COMPLETIONS_API, LOCAL_TOKEN, anthropicLocalModelId, codexLocalModelId, harnessEndpoints } from "./shared"

export type SetupPlatform = "darwin" | "linux" | "win32"

/**
 * What a setup prompt is rendered for. `origin` is Magnitude as the harness's machine reaches it,
 * `key` is absent when that machine is the one running Magnitude, and `updatedAt` stamps settings a
 * harness only keeps when they carry a write time.
 */
export interface HarnessSetupTarget {
  readonly harness: HarnessId
  readonly models: ReadonlyArray<HarnessModel>
  readonly model: HarnessModel
  readonly platform: SetupPlatform
  readonly origin: string
  readonly key: Option.Option<string>
  readonly updatedAt: string
}

export interface HarnessSetup {
  readonly prompt: string
  readonly runCommand: string
  readonly docsUrl: string
}

type Format = "json" | "json5" | "yaml" | "toml"

interface SetupFile {
  readonly path: string
  readonly format: Format
  readonly content: unknown
  readonly purpose: string
}

interface SetupPlan {
  readonly name: string
  readonly files: ReadonlyArray<SetupFile>
  readonly overrides: ReadonlyArray<string>
  readonly environment: ReadonlyArray<{ readonly name: string; readonly value: string }>
  readonly notes: ReadonlyArray<string>
}

const render = (format: Format, content: unknown): string => {
  switch (format) {
    case "json":
    case "json5": return JSON.stringify(content, null, 2)
    case "yaml": return stringifyYaml(content).trimEnd()
    case "toml": return stringifyToml(content as Record<string, unknown>).trimEnd()
  }
}

const homePath = (platform: SetupPlatform, relative: string) => platform === "win32"
  ? `%USERPROFILE%\\${relative.replaceAll("/", "\\")}`
  : `~/${relative}`

const plan = (target: HarnessSetupTarget): SetupPlan => {
  const { model, models, platform } = target
  const endpoints = harnessEndpoints(target.origin)
  const apiKey = Option.getOrElse(target.key, () => LOCAL_TOKEN)
  const at = (relative: string) => homePath(platform, relative)
  switch (Brand.unbranded(target.harness)) {
    case "pi": return {
      name: "Pi",
      files: [
        { path: at(".pi/agent/models.json"), format: "json", purpose: "the Magnitude provider",
          content: { providers: { magnitude: { ...piProviderConfig(models, endpoints.openai), apiKey } } } },
        { path: at(".pi/agent/settings.json"), format: "json", purpose: "make Magnitude the default",
          content: { defaultProvider: "magnitude", defaultModel: model.id } },
      ],
      overrides: ["PI_CODING_AGENT_DIR"],
      environment: [],
      notes: [],
    }
    case "opencode": {
      const provider = openCodeProviderConfig(models, endpoints.openai)
      return {
        name: "OpenCode",
        files: [{ path: at(".config/opencode/opencode.json"), format: "json", purpose: "the Magnitude provider and default model",
          content: { provider: { magnitude: { ...provider, options: { ...provider.options, apiKey } } }, model: `magnitude/${model.id}` } }],
        overrides: ["OPENCODE_CONFIG", "XDG_CONFIG_HOME"],
        environment: [],
        notes: [],
      }
    }
    case "hermes": return {
      name: "Hermes",
      files: [{ path: at(".hermes/config.yaml"), format: "yaml", purpose: "the Magnitude provider, default model, and reasoning setting",
        content: {
          providers: { magnitude: { ...hermesProviderConfig(endpoints.openai), api_key: apiKey } },
          model: { provider: "custom:magnitude", default: model.id },
          agent: { reasoning_overrides: hermesReasoningOverrides(models) },
        } }],
      overrides: ["HERMES_HOME"],
      environment: [],
      notes: [],
    }
    case "openclaw": return {
      name: "OpenClaw",
      files: [{ path: at(".openclaw/openclaw.json"), format: "json5", purpose: "the Magnitude provider and default agent model",
        content: {
          models: { providers: { magnitude: { baseUrl: endpoints.openai, apiKey, api: CHAT_COMPLETIONS_API, models: openClawModels(models) } } },
          agents: { defaults: { model: { primary: `magnitude/${model.id}` } }, entries: { magnitude: openClawAgentConfig(model) } },
        } }],
      overrides: ["OPENCLAW_CONFIG_PATH", "OPENCLAW_STATE_DIR"],
      environment: [],
      notes: ["OpenClaw selects its model inside its TUI; choose the Magnitude model there if it isn't already selected."],
    }
    case "codex": return {
      name: "Codex",
      files: [{ path: at(".codex/config.toml"), format: "toml", purpose: "the Magnitude provider and default model",
        content: {
          model_provider: "magnitude",
          model: codexLocalModelId(model.id),
          model_context_window: model.contextWindow,
          model_max_output_tokens: modelMaxTokens(model),
          model_providers: { magnitude: { name: "Magnitude", base_url: endpoints.codex, wire_api: "responses", env_key: "MAGNITUDE_API_KEY" } },
        } }],
      overrides: ["CODEX_HOME"],
      environment: [{ name: "MAGNITUDE_API_KEY", value: apiKey }],
      notes: [
        "Keep any existing top-level keys other than model_provider and model, and any other [model_providers.*] tables.",
        "Codex may warn that it has no metadata for this model; that warning is expected and harmless.",
        "Only Magnitude's local models are available through this provider.",
      ],
    }
    case "claude-code": return {
      name: "Claude Code",
      files: [{ path: at(".claude/settings.json"), format: "json", purpose: "route Claude Code through Magnitude",
        content: {
          env: {
            ANTHROPIC_BASE_URL: endpoints.anthropic,
            ...(Option.isSome(target.key) ? { ANTHROPIC_AUTH_TOKEN: target.key.value } : {}),
            CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY: CLAUDE_GATEWAY_DISCOVERY,
          },
          model: anthropicLocalModelId(model.id),
        } }],
      overrides: ["CLAUDE_CONFIG_DIR"],
      environment: [],
      notes: Option.isSome(target.key)
        ? ["Only Magnitude's local models are available from this computer; Claude's own models need Claude Code's normal setup."]
        : ["Claude's own models keep working through your existing Claude login; Magnitude's local models appear as anthropic-local/…"],
    }
    case "oh-my-pi": return {
      name: "Oh My Pi",
      files: [
        { path: at(".omp/agent/models.yml"), format: "yaml", purpose: "the Magnitude provider",
          content: { providers: { magnitude: { baseUrl: endpoints.openai, apiKey, api: CHAT_COMPLETIONS_API, models: ohMyPiModels(models) } } } },
        { path: at(".omp/agent/config.yml"), format: "yaml", purpose: "make Magnitude the default model",
          content: { modelRoles: { default: `magnitude/${model.id}` } } },
      ],
      overrides: ["PI_CODING_AGENT_DIR", "PI_CONFIG_DIR", "OMP_PROFILE"],
      environment: [],
      notes: [],
    }
    case "cline": return {
      name: "Cline",
      files: [
        { path: at(".cline/data/settings/providers.json"), format: "json", purpose: "the Magnitude provider",
          content: {
            version: 1,
            modes: {},
            providers: { "openai-compatible": { settings: { ...clineProviderSettings(Option.some(model.id), endpoints.openai), apiKey }, updatedAt: target.updatedAt, tokenSource: "manual" } },
            lastUsedProvider: "openai-compatible",
          } },
        { path: at(".cline/data/settings/models.json"), format: "json", purpose: "the model's capabilities",
          content: { version: 1, providers: { "openai-compatible": clineModelRegistryEntry(models) } } },
      ],
      overrides: ["CLINE_DATA_DIR", "CLINE_PROVIDER_SETTINGS_PATH"],
      environment: [],
      notes: [],
    }
  }
}

const capabilities = (model: HarnessModel) => [
  `context window ${model.contextWindow.toLocaleString("en-US")} tokens`,
  `up to ${modelMaxTokens(model).toLocaleString("en-US")} output tokens`,
  model.capabilities.reasoning.supported ? `reasoning (${model.capabilities.reasoning.efforts.join(", ")})` : "no reasoning",
  model.capabilities.vision ? "image input" : "text input only",
].join(", ")

const environmentStep = (platform: SetupPlatform, name: string, value: string) => platform === "win32"
  ? `Set the user environment variable ${name} to ${value} so it is present whenever the harness starts (for example \`[Environment]::SetEnvironmentVariable("${name}", "${value}", "User")\`), then open a new terminal.`
  : `Export ${name}=${value} from the shell profile this user's terminals load (for example ~/.zshrc or ~/.bashrc), then open a new terminal.`

const checkCommand = (platform: SetupPlatform, origin: string, key: Option.Option<string>) => {
  const url = `${harnessEndpoints(origin).openai}/models`
  return Option.match(key, {
    onNone: () => `${platform === "win32" ? "curl.exe" : "curl"} -s ${url}`,
    onSome: (value) => `${platform === "win32" ? "curl.exe" : "curl"} -s -H "Authorization: Bearer ${value}" ${url}`,
  })
}

/**
 * A prompt a person pastes into the harness itself (or any coding agent) on the computer that will
 * run it. The configuration comes from the same builders the one-click connectors write, so both
 * paths describe one setup; the agent merges it into whatever that computer already has.
 */
export const describeHarnessSetup = (target: HarnessSetupTarget): HarnessSetup => {
  const setup = plan(target)
  const runCommand = harnessCommand(target.harness, target.model.id, target.platform)
  const files = setup.files.map((file, index) => [
    `${index + 1}. ${file.path} — ${file.purpose}`,
    "```" + file.format,
    render(file.format, file.content),
    "```",
  ].join("\n"))
  const prompt = [
    `Set up ${setup.name} on this computer to use Magnitude, an inference server for local models. If ${setup.name} isn't installed yet, install it first (${harnessInstallationUrl(target.harness)}).`,
    "",
    `Server: ${harnessEndpoints(target.origin).openai.replace(/\/v1$/, "")}`,
    `API key: ${Option.getOrElse(target.key, () => `${LOCAL_TOKEN} (any value works on this computer)`)}`,
    `Default model: ${target.model.id} — ${target.model.name}; ${capabilities(target.model)}`,
    "",
    `Merge the settings below into ${setup.name}'s configuration files. Keep every existing provider, model, and setting that isn't shown here; replace only the Magnitude entries and the default-model keys shown. Create a file or its folders only if they don't exist yet. Use the same file format and keep comments. If ${setup.overrides.join(" or ")} is set, use the location it points to instead of the default path.`,
    `After editing, check that each file still parses as ${setup.files.map(file => file.format.toUpperCase()).filter((format, index, all) => all.indexOf(format) === index).join(" or ")}.`,
    "",
    ...files.flatMap(file => [file, ""]),
    ...setup.environment.map(({ name, value }) => `${environmentStep(target.platform, name, value)}\n`),
    ...(setup.notes.length > 0 ? [setup.notes.map(note => `- ${note}`).join("\n"), ""] : []),
    `Then confirm the server answers from this computer: \`${checkCommand(target.platform, target.origin, target.key)}\` should list ${target.model.id}.`,
    `Finally, tell me ${setup.name} is ready and that I can start it from my project folder with: ${runCommand}`,
  ].join("\n")
  return { prompt, runCommand, docsUrl: `https://docs.magnitude.dev/integrations/${target.harness}` }
}
