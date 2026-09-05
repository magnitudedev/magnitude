import { homedir } from "node:os"
import { resolve } from "node:path"

export type SkillInstallationTarget = "shared-agents" | "hermes-user" | "claude-user" | "cline-user"

export interface SkillInstallationPaths {
  readonly skillFile: string
}

export interface HarnessConnectionPaths {
  readonly manifest: string
  readonly piModels: string
  readonly piSettings: string
  readonly opencode: string
  readonly hermes: string
  readonly openclaw: string
  readonly codex: string
  readonly codexUser: string
  readonly codexModels: string
  readonly claude: string
  readonly ompModels: string
  readonly ompSettings: string
  readonly clineProviders: string
  readonly clineModels: string
  readonly gptme: string
  readonly skillInstallations: Readonly<Record<SkillInstallationTarget, SkillInstallationPaths>>
}

export const harnessConnectionPaths = (): HarnessConnectionPaths => {
  const home = homedir()
  const piRoot = resolve(process.env.PI_CODING_AGENT_DIR ?? `${home}/.pi/agent`)
  const clineRoot = `${home}/.cline/data`
  const hermesRoot = process.env.HERMES_HOME ?? `${home}/.hermes`
  const openClawRoot = process.env.OPENCLAW_STATE_DIR ?? `${home}/.openclaw`
  const codexRoot = process.env.CODEX_HOME ?? `${home}/.codex`
  const skillInstallation = (root: string): SkillInstallationPaths => ({
    skillFile: `${root}/magnitude/SKILL.md`,
  })
  return {
    manifest: `${home}/.magnitude/harness-connections.json`,
    piModels: `${piRoot}/models.json`,
    piSettings: `${piRoot}/settings.json`,
    opencode: `${home}/.config/opencode/opencode.json`,
    hermes: `${hermesRoot}/config.yaml`,
    openclaw: `${openClawRoot}/openclaw.json`,
    codex: `${codexRoot}/magnitude.config.toml`,
    codexUser: `${codexRoot}/config.toml`,
    codexModels: `${codexRoot}/magnitude.models.json`,
    claude: `${process.env.CLAUDE_CONFIG_DIR ?? `${home}/.claude`}/settings.json`,
    ompModels: `${home}/.omp/agent/models.yml`,
    ompSettings: `${home}/.omp/agent/config.yml`,
    clineProviders: `${clineRoot}/settings/providers.json`,
    clineModels: `${clineRoot}/settings/models.json`,
    gptme: `${home}/.config/gptme/config.toml`,
    skillInstallations: {
      "shared-agents": skillInstallation(`${home}/.agents/skills`),
      "hermes-user": skillInstallation(`${hermesRoot}/skills`),
      "claude-user": skillInstallation(`${process.env.CLAUDE_CONFIG_DIR ?? `${home}/.claude`}/skills`),
      "cline-user": skillInstallation(`${clineRoot}/settings/skills`),
    },
  }
}
