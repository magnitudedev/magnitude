import { homedir } from "node:os"

export interface HarnessConnectionPaths {
  readonly manifest: string
  readonly piModels: string
  readonly piSettings: string
  readonly opencode: string
  readonly hermes: string
  readonly openclaw: string
  readonly codex: string
  readonly codexModels: string
  readonly claude: string
  readonly ompModels: string
  readonly ompSettings: string
  readonly clineProviders: string
  readonly clineModels: string
  readonly skills: Readonly<Record<string, string>>
}

export const harnessConnectionPaths = (): HarnessConnectionPaths => {
  const home = homedir()
  const clineRoot = process.env.CLINE_DATA_DIR ?? `${home}/.cline/data`
  const hermesRoot = process.env.HERMES_HOME ?? `${home}/.hermes`
  const openClawRoot = process.env.OPENCLAW_STATE_DIR ?? `${home}/.openclaw`
  const codexRoot = process.env.CODEX_HOME ?? `${home}/.codex`
  return {
    manifest: `${home}/.magnitude/harness-connections.json`,
    piModels: `${home}/.pi/agent/models.json`,
    piSettings: `${home}/.pi/agent/settings.json`,
    opencode: `${home}/.config/opencode/opencode.json`,
    hermes: `${hermesRoot}/config.yaml`,
    openclaw: `${openClawRoot}/openclaw.json`,
    codex: `${codexRoot}/magnitude.config.toml`,
    codexModels: `${codexRoot}/magnitude.models.json`,
    claude: `${process.env.CLAUDE_CONFIG_DIR ?? `${home}/.claude`}/settings.json`,
    ompModels: `${home}/.omp/agent/models.yml`,
    ompSettings: `${home}/.omp/agent/settings.json`,
    clineProviders: `${clineRoot}/settings/providers.json`,
    clineModels: `${clineRoot}/settings/models.json`,
    skills: {
      magnitude: `${home}/.magnitude/skills/magnitude/SKILL.md`,
      pi: `${home}/.pi/agent/skills/magnitude/SKILL.md`,
      opencode: `${home}/.config/opencode/skills/magnitude/SKILL.md`,
      hermes: `${hermesRoot}/skills/magnitude/SKILL.md`,
      openclaw: `${openClawRoot}/skills/magnitude/SKILL.md`,
      codex: `${home}/.agents/skills/magnitude/SKILL.md`,
      "claude-code": `${home}/.claude/skills/magnitude/SKILL.md`,
      "oh-my-pi": `${home}/.omp/agent/skills/magnitude/SKILL.md`,
      cline: `${clineRoot}/settings/skills/magnitude/SKILL.md`,
    },
  }
}
