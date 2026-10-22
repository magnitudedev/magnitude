import { HarnessConnectionError } from "./types"
import { FileSystem } from "@effect/platform"
import { Effect } from "effect"
import { createHash } from "node:crypto"
import { homedir, userInfo } from "node:os"
import { dirname, resolve } from "node:path"

export type SkillInstallationTarget = "shared-agents" | "hermes-user" | "claude-user" | "cline-user"

export interface SkillInstallationPaths {
  readonly skillFile: string
}

export interface KeychainItem {
  readonly service: string
  readonly account: string
}

/** Where Claude Code keeps a login of its own; only the existence of a credential is ever read. */
export interface ClaudeCodeLoginLocations {
  readonly credentialFiles: ReadonlyArray<string>
  readonly globalConfigFiles: ReadonlyArray<string>
  readonly keychainItems: ReadonlyArray<KeychainItem>
}

export interface HarnessConnectionPaths {
  readonly manifest: string
  readonly piModels: string
  readonly piSettings: string
  readonly opencodeFiles?: ReadonlyArray<string>
  readonly opencode: string
  readonly hermes: string
  readonly openclaw: string
  readonly codex: string
  readonly codexUser: string
  readonly codexModels: string
  readonly claude: string
  readonly claudeLogin: ClaudeCodeLoginLocations
  readonly ompModels: string
  readonly ompSettings: string
  readonly clineProviders: string
  readonly clineModels: string
  readonly skillInstallations: Readonly<Record<SkillInstallationTarget, SkillInstallationPaths>>
}

const keychainAccount = (environment: Readonly<Record<string, string | undefined>>): string => {
  let account: string
  try {
    account = environment.USER || userInfo().username
  } catch {
    return "claude-code-user"
  }
  return /^[a-zA-Z0-9._-]+$/.test(account) ? account : "claude-code-user"
}

/**
 * Mirrors Claude Code's own storage: OAuth credentials in `.credentials.json` under its configuration
 * directory (the macOS Keychain first on a Mac), and a Console API key in the Keychain or the global
 * config's `primaryApiKey`. A custom `CLAUDE_CONFIG_DIR` suffixes Keychain services with its hash.
 */
const claudeCodeLoginLocations = (
  home: string,
  environment: Readonly<Record<string, string | undefined>>,
  keychain: boolean,
): ClaudeCodeLoginLocations => {
  const configDirectory = (environment.CLAUDE_CONFIG_DIR ?? `${home}/.claude`).normalize("NFC")
  const suffix = environment.CLAUDE_CONFIG_DIR
    ? `-${createHash("sha256").update(configDirectory).digest("hex").substring(0, 8)}`
    : ""
  const account = keychainAccount(environment)
  return {
    credentialFiles: [`${configDirectory}/.credentials.json`],
    globalConfigFiles: [`${configDirectory}/.config.json`, `${environment.CLAUDE_CONFIG_DIR || home}/.claude.json`],
    keychainItems: keychain
      ? [`Claude Code-credentials${suffix}`, `Claude Code${suffix}`].map((service) => ({ service, account }))
      : [],
  }
}

export const harnessConnectionPaths = (
  isolatedHome?: string,
  inheritedEnvironment: Readonly<Record<string, string | undefined>> = process.env,
  platform: NodeJS.Platform = process.platform,
): HarnessConnectionPaths => {
  const home = isolatedHome ?? homedir()
  const environment = isolatedHome === undefined ? inheritedEnvironment : {}
  const piRoot = resolve(environment.PI_CODING_AGENT_DIR ?? `${home}/.pi/agent`)
  const ompProfile = (environment.OMP_PROFILE ?? environment.PI_PROFILE)?.trim()
  const namedOmpProfile = ompProfile && ompProfile !== "default" ? ompProfile : undefined
  if (namedOmpProfile && (!/^[a-z0-9][a-z0-9._-]{0,63}$/.test(namedOmpProfile)
      || namedOmpProfile.endsWith(".") || /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?$/i.test(namedOmpProfile))) {
    throw new Error("Invalid Oh My Pi profile name")
  }
  const ompConfigRoot = `${home}/${environment.PI_CONFIG_DIR || ".omp"}`
  const ompRoot = resolve(namedOmpProfile ? `${ompConfigRoot}/profiles/${namedOmpProfile}/agent`
    : environment.PI_CODING_AGENT_DIR || `${ompConfigRoot}/agent`)
  const clineRoot = environment.CLINE_DATA_DIR || `${home}/.cline/data`
  const clineProviders = environment.CLINE_PROVIDER_SETTINGS_PATH || `${clineRoot}/settings/providers.json`
  // Native Windows Hermes keeps its home in the local application data folder, not the profile.
  const hermesRoot = environment.HERMES_HOME ?? (platform === "win32"
    ? `${environment.LOCALAPPDATA?.trim() || `${home}/AppData/Local`}/hermes`
    : `${home}/.hermes`)
  const openClawRoot = environment.OPENCLAW_STATE_DIR ?? `${home}/.openclaw`
  const codexRoot = environment.CODEX_HOME ?? `${home}/.codex`
  const skillInstallation = (root: string): SkillInstallationPaths => ({
    skillFile: `${root}/magnitude/SKILL.md`,
  })
  return {
    manifest: `${home}/.magnitude/harness-connections.json`,
    piModels: `${piRoot}/models.json`,
    piSettings: `${piRoot}/settings.json`,
    opencode: environment.OPENCODE_CONFIG || `${environment.XDG_CONFIG_HOME || `${home}/.config`}/opencode/opencode.json`,
    hermes: `${hermesRoot}/config.yaml`,
    openclaw: environment.OPENCLAW_CONFIG_PATH || `${openClawRoot}/openclaw.json`,
    codex: `${codexRoot}/magnitude.config.toml`,
    codexUser: `${codexRoot}/config.toml`,
    codexModels: `${codexRoot}/magnitude.models.json`,
    claude: `${environment.CLAUDE_CONFIG_DIR ?? `${home}/.claude`}/settings.json`,
    claudeLogin: claudeCodeLoginLocations(home, environment, isolatedHome === undefined && process.platform === "darwin"),
    ompModels: `${ompRoot}/models.yml`,
    ompSettings: `${ompRoot}/config.yml`,
    clineProviders,
    clineModels: `${dirname(clineProviders)}/models.json`,
    skillInstallations: {
      "shared-agents": skillInstallation(`${home}/.agents/skills`),
      "hermes-user": skillInstallation(`${hermesRoot}/skills`),
      "claude-user": skillInstallation(`${environment.CLAUDE_CONFIG_DIR ?? `${home}/.claude`}/skills`),
      "cline-user": skillInstallation(`${clineRoot}/settings/skills`),
    },
  }
}

/** OpenCode merges global files in order, then the explicit override. */
export const resolveHarnessConnectionPaths = (isolatedHome?: string, environment: Readonly<Record<string, string | undefined>> = process.env) => Effect.gen(function* () {
  const paths = yield* Effect.try({
    try: () => harnessConnectionPaths(isolatedHome, environment),
    catch: error => new HarnessConnectionError({ operation: "list", message: String(error) }),
  })
  const fs = yield* FileSystem.FileSystem
  const globalRoot = `${isolatedHome === undefined ? environment.XDG_CONFIG_HOME || `${homedir()}/.config` : `${isolatedHome}/.config`}/opencode`
  const candidates = [`${globalRoot}/config.json`, `${globalRoot}/opencode.json`, `${globalRoot}/opencode.jsonc`]
  const explicit = isolatedHome === undefined ? environment.OPENCODE_CONFIG : undefined
  const existing = yield* Effect.filter(candidates, path => fs.exists(path))
  const target = explicit || existing.at(-1) || paths.opencode
  const files = [...new Set([...existing.filter(path => path !== target), target])]
  return { ...paths, opencode: target, opencodeFiles: files }
}).pipe(Effect.mapError(error => error instanceof HarnessConnectionError ? error
  : new HarnessConnectionError({ operation: "list", message: `Could not resolve harness configuration: ${error.message}` })))
