import { Brand, Option, Schema } from "effect"
import type { ProviderModelId } from "@magnitudedev/ai/provider/model"
import type { HarnessId } from "./schemas/harness-connections"

/** Commands are pasted into a POSIX shell on macOS/Linux or PowerShell on Windows. */
export const shellArgument = (value: string, platform: string): string => /^[a-zA-Z0-9_./:=+@,-]+$/.test(value)
  ? value
  : `'${value.replaceAll("'", platform === "win32" ? "''" : "'\\''")}'`

const encodeOpenCodeModelConfiguration = Schema.encodeSync(Schema.parseJson(Schema.Struct({ model: Schema.String })))

/** OpenCode 1.x, when the installed version says so; an unknown version is assumed current. */
const isOpenCodeV1 = (version: Option.Option<string>) => Option.exists(version, (value) => /^1\./.test(value))

/**
 * OpenCode 2's TUI dropped `--model` and attaches to a shared background service that keeps the configuration
 * of whichever launch started it, so the model is pinned with inline configuration on a private `--standalone` server.
 */
const openCodeV2Command = (model: ProviderModelId, platform: string): string => {
  const configuration = shellArgument(encodeOpenCodeModelConfiguration({ model: `magnitude/${model}` }), platform)
  return platform === "win32"
    ? `$env:OPENCODE_CONFIG_CONTENT=${configuration}; opencode --standalone`
    : `OPENCODE_CONFIG_CONTENT=${configuration} opencode --standalone`
}

export const harnessCommand = (harness: HarnessId, model: ProviderModelId, platform: string, version: Option.Option<string>): string => {
  const argv = (() => {
    switch (Brand.unbranded(harness)) {
      case "pi": return ["pi", "--provider", "magnitude", "--model", model]
      case "opencode": return isOpenCodeV1(version) ? ["opencode", "--model", `magnitude/${model}`] : undefined
      case "hermes": return ["hermes", "--provider", "custom:magnitude", "--model", model]
      case "codex": return ["codex", "-c", "model_provider=magnitude", "--model", `magnitude-local/${model}`]
      case "claude-code": return ["claude", "--model", `anthropic-local/${model}`]
      case "oh-my-pi": return ["omp", "--model", `magnitude/${model}`]
      case "cline": return ["cline", "--tui", "--provider", "openai-compatible", "--model", model]
      // OpenClaw selects the model inside its TUI, not through a launch flag.
      case "openclaw": return ["openclaw", "tui"]
    }
  })()
  return argv === undefined ? openCodeV2Command(model, platform) : argv.map(arg => shellArgument(arg, platform)).join(" ")
}

const installationUrls: Record<Brand.Brand.Unbranded<HarnessId>, string> = {
  pi: "https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent#quick-start",
  opencode: "https://opencode.ai/docs/#install",
  hermes: "https://hermes-agent.nousresearch.com/docs/getting-started/installation/",
  openclaw: "https://docs.openclaw.ai/install",
  codex: "https://developers.openai.com/codex/cli",
  "claude-code": "https://code.claude.com/docs/en/overview",
  "oh-my-pi": "https://github.com/can1357/oh-my-pi#installation",
  cline: "https://docs.cline.bot/getting-started/installing-cline",
}

/** Where a person installs a harness that isn't on their computer yet. */
export const harnessInstallationUrl = (harness: HarnessId): string => installationUrls[Brand.unbranded(harness)]
