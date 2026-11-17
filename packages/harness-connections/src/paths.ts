import { HarnessConnectionError } from "./types"
import { FileSystem } from "@effect/platform"
import { Effect } from "effect"
import { createHash } from "node:crypto"
import { homedir, userInfo } from "node:os"
import { posix, win32, type PlatformPath } from "node:path"

/** Path functions for the platform the agents run on, so each path reads and opens with its own separators. */
const pathsFor = (platform: NodeJS.Platform): PlatformPath => platform === "win32" ? win32 : posix

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
  path: PlatformPath,
): ClaudeCodeLoginLocations => {
  const configDirectory = (environment.CLAUDE_CONFIG_DIR ?? path.join(home, ".claude")).normalize("NFC")
  const suffix = environment.CLAUDE_CONFIG_DIR
    ? `-${createHash("sha256").update(configDirectory).digest("hex").substring(0, 8)}`
    : ""
  const account = keychainAccount(environment)
  return {
    credentialFiles: [path.join(configDirectory, ".credentials.json")],
    globalConfigFiles: [path.join(configDirectory, ".config.json"), path.join(environment.CLAUDE_CONFIG_DIR || home, ".claude.json")],
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
  const path = pathsFor(platform)
  const home = isolatedHome ?? homedir()
  const environment = isolatedHome === undefined ? inheritedEnvironment : {}
  const piRoot = path.resolve(environment.PI_CODING_AGENT_DIR ?? path.join(home, ".pi", "agent"))
  const ompProfile = (environment.OMP_PROFILE ?? environment.PI_PROFILE)?.trim()
  const namedOmpProfile = ompProfile && ompProfile !== "default" ? ompProfile : undefined
  if (namedOmpProfile && (!/^[a-z0-9][a-z0-9._-]{0,63}$/.test(namedOmpProfile)
      || namedOmpProfile.endsWith(".") || /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?$/i.test(namedOmpProfile))) {
    throw new Error("Invalid Oh My Pi profile name")
  }
  const ompConfigRoot = path.join(home, environment.PI_CONFIG_DIR || ".omp")
  const ompRoot = path.resolve(namedOmpProfile ? path.join(ompConfigRoot, "profiles", namedOmpProfile, "agent")
    : environment.PI_CODING_AGENT_DIR || path.join(ompConfigRoot, "agent"))
  const clineRoot = environment.CLINE_DATA_DIR || path.join(home, ".cline", "data")
  const clineProviders = environment.CLINE_PROVIDER_SETTINGS_PATH || path.join(clineRoot, "settings", "providers.json")
  // Native Windows Hermes keeps its home in the local application data folder, not the profile.
  const hermesRoot = environment.HERMES_HOME ?? (platform === "win32"
    ? path.join(environment.LOCALAPPDATA?.trim() || path.join(home, "AppData", "Local"), "hermes")
    : path.join(home, ".hermes"))
  const openClawRoot = environment.OPENCLAW_STATE_DIR ?? path.join(home, ".openclaw")
  const codexRoot = environment.CODEX_HOME ?? path.join(home, ".codex")
  const claudeRoot = environment.CLAUDE_CONFIG_DIR ?? path.join(home, ".claude")
  const skillInstallation = (root: string): SkillInstallationPaths => ({
    skillFile: path.join(root, "magnitude", "SKILL.md"),
  })
  return {
    manifest: path.join(home, ".magnitude", "harness-connections.json"),
    piModels: path.join(piRoot, "models.json"),
    piSettings: path.join(piRoot, "settings.json"),
    opencode: environment.OPENCODE_CONFIG || path.join(environment.XDG_CONFIG_HOME || path.join(home, ".config"), "opencode", "opencode.json"),
    hermes: path.join(hermesRoot, "config.yaml"),
    openclaw: environment.OPENCLAW_CONFIG_PATH || path.join(openClawRoot, "openclaw.json"),
    codex: path.join(codexRoot, "magnitude.config.toml"),
    codexUser: path.join(codexRoot, "config.toml"),
    codexModels: path.join(codexRoot, "magnitude.models.json"),
    claude: path.join(claudeRoot, "settings.json"),
    claudeLogin: claudeCodeLoginLocations(home, environment, isolatedHome === undefined && process.platform === "darwin", path),
    ompModels: path.join(ompRoot, "models.yml"),
    ompSettings: path.join(ompRoot, "config.yml"),
    clineProviders,
    clineModels: path.join(path.dirname(clineProviders), "models.json"),
    skillInstallations: {
      "shared-agents": skillInstallation(path.join(home, ".agents", "skills")),
      "hermes-user": skillInstallation(path.join(hermesRoot, "skills")),
      "claude-user": skillInstallation(path.join(claudeRoot, "skills")),
      "cline-user": skillInstallation(path.join(clineRoot, "settings", "skills")),
    },
  }
}

/** OpenCode merges global files in order, then the explicit override. */
export const resolveHarnessConnectionPaths = (
  isolatedHome?: string,
  environment: Readonly<Record<string, string | undefined>> = process.env,
  platform: NodeJS.Platform = process.platform,
) => Effect.gen(function* () {
  const path = pathsFor(platform)
  const paths = yield* Effect.try({
    try: () => harnessConnectionPaths(isolatedHome, environment, platform),
    catch: error => new HarnessConnectionError({ operation: "list", message: String(error) }),
  })
  const fs = yield* FileSystem.FileSystem
  const globalRoot = path.join(isolatedHome === undefined ? environment.XDG_CONFIG_HOME || path.join(homedir(), ".config") : path.join(isolatedHome, ".config"), "opencode")
  const candidates = [path.join(globalRoot, "config.json"), path.join(globalRoot, "opencode.json"), path.join(globalRoot, "opencode.jsonc")]
  const explicit = isolatedHome === undefined ? environment.OPENCODE_CONFIG : undefined
  const existing = yield* Effect.filter(candidates, candidate => fs.exists(candidate))
  const target = explicit || existing.at(-1) || paths.opencode
  const files = [...new Set([...existing.filter(candidate => candidate !== target), target])]
  return { ...paths, opencode: target, opencodeFiles: files }
}).pipe(Effect.mapError(error => error instanceof HarnessConnectionError ? error
  : new HarnessConnectionError({ operation: "list", message: `Could not resolve harness configuration: ${error.message}` })))
