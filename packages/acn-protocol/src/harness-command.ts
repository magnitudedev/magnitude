import { Brand } from "effect"
import type { ProviderModelId } from "@magnitudedev/ai/provider/model"
import type { HarnessId } from "./schemas/harness-connections"

/** Commands are pasted into a POSIX shell on macOS/Linux or PowerShell on Windows. */
export const shellArgument = (value: string, platform: string): string => /^[a-zA-Z0-9_./:=+@,-]+$/.test(value)
  ? value
  : `'${value.replaceAll("'", platform === "win32" ? "''" : "'\\''")}'`

export const harnessCommand = (harness: HarnessId, model: ProviderModelId, platform: string): string => {
  const argv = (() => {
    switch (Brand.unbranded(harness)) {
      case "pi": return ["pi", "--provider", "magnitude", "--model", model]
      case "opencode": return ["opencode", "--model", `magnitude/${model}`]
      case "hermes": return ["hermes", "--provider", "custom:magnitude", "--model", model]
      case "codex": return ["codex", "-c", "model_provider=magnitude", "--model", `magnitude-local/${model}`]
      case "claude-code": return ["claude", "--model", `anthropic-local/${model}`]
      case "oh-my-pi": return ["omp", "--model", `magnitude/${model}`]
      case "cline": return ["cline", "--tui", "--provider", "openai-compatible", "--model", model]
      // OpenClaw selects the model inside its TUI, not through a launch flag.
      case "openclaw": return ["openclaw", "tui"]
    }
  })()
  return argv.map(arg => shellArgument(arg, platform)).join(" ")
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
