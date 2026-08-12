import { getAllCommands, type SlashCommandDefinition } from './slash-commands'

/** Optional feature-flag lookup function. Defaults to checking process.env. */
export type GetFeatureFlag = (key: string) => boolean | undefined
export type ModelMenuId = "models" | "catalog" | "hardware" | "cloud"

const defaultGetFeatureFlag: GetFeatureFlag = (key) => {
  if (typeof process !== 'undefined' && process.env) {
    return !!process.env[key]
  }
  return undefined
}

let featureFlagFn: GetFeatureFlag = defaultGetFeatureFlag

/** Override the feature-flag lookup (e.g. for non-Node environments like Electron). */
export function setFeatureFlagLookup(fn: GetFeatureFlag | undefined): void {
  featureFlagFn = fn ?? defaultGetFeatureFlag
}

function isCommandAvailable(cmd: SlashCommandDefinition): boolean {
  if (!cmd.featureFlag) return true
  return !!featureFlagFn(cmd.featureFlag)
}

function getAvailableCommands(): SlashCommandDefinition[] {
  return getAllCommands().filter(isCommandAvailable)
}

/** Context provided to command handlers by app.tsx */
export interface CommandContext {
  /** Reset the conversation: dispose current client, clear display, create new client */
  resetConversation: () => void
  /** Show a system/info message in the chat (not sent to agent) */
  showSystemMessage: (message: string) => void
  /** Exit the application */
  exitApp: () => void
  /** Open the recent chats overlay */
  openRecentChats: () => void
  /** Enter bash mode for running terminal commands */
  enterBashMode: () => void
  /** Run the /init flow: explore codebase and generate AGENTS.md */
  initProject: () => void
  /** Legacy settings entry point for clients without the model-menu surface. */
  openSettings: () => void
  /** Open the usage overlay */
  openUsage: () => void
  /** Open the client-owned cloud model setup surface, when available. */
  openCloud?: () => void
  /** Open a client-owned model menu root, when available. */
  openModelMenu?: (menu: ModelMenuId) => void
  /** Toggle between the default and transcript timeline presentations. */
  toggleTranscript?: () => void
  /** Toggle autopilot mode */
  toggleAutopilot: () => void
}

/**
 * Parse input text to detect a slash command.
 * Returns the matched command id and any arguments, or null if not a command.
 *
 * - Input must start with '/'
 * - Command token is the first whitespace-delimited word (without the '/')
 * - Matches against command id or aliases (case-insensitive)
 */
export function parseSlashCommand(input: string): { commandId: string; args: string } | null {
  const trimmed = input.trim()
  if (!trimmed.startsWith('/')) return null

  const withoutSlash = trimmed.slice(1)
  const spaceIndex = withoutSlash.indexOf(' ')
  const commandToken = (spaceIndex === -1 ? withoutSlash : withoutSlash.slice(0, spaceIndex)).toLowerCase()
  const args = spaceIndex === -1 ? '' : withoutSlash.slice(spaceIndex + 1).trim()

  if (!commandToken) return null

  for (const cmd of getAvailableCommands()) {
    if (cmd.id === commandToken) {
      return { commandId: cmd.id, args }
    }
    if (cmd.aliases?.some(alias => alias === commandToken)) {
      return { commandId: cmd.id, args }
    }
  }

  return null
}

/**
 * Filter slash commands by query string (text after '/').
 * Returns matching commands sorted by: prefix matches first, then substring matches.
 * Empty query returns all commands.
 */
export function filterSlashCommands(query: string): SlashCommandDefinition[] {
  if (!query) return [...getAvailableCommands()]

  const lowerQuery = query.toLowerCase()
  const prefixMatches: SlashCommandDefinition[] = []
  const substringMatches: SlashCommandDefinition[] = []

  for (const cmd of getAvailableCommands()) {
    const matchesId = cmd.id.toLowerCase().startsWith(lowerQuery)
    const matchesAlias = cmd.aliases?.some(a => a.toLowerCase().startsWith(lowerQuery))

    if (matchesId || matchesAlias) {
      prefixMatches.push(cmd)
      continue
    }

    const substringId = cmd.id.toLowerCase().includes(lowerQuery)
    const substringAlias = cmd.aliases?.some(a => a.toLowerCase().includes(lowerQuery))
    const substringDesc = cmd.description.toLowerCase().includes(lowerQuery)

    if (substringId || substringAlias || substringDesc) {
      substringMatches.push(cmd)
    }
  }

  return [...prefixMatches, ...substringMatches]
}

/**
 * Route user input through slash command handlers.
 *
 * @returns true if the input was handled as a slash command, false if it
 *          should be passed through to the agent as a normal message.
 */
export function routeSlashCommand(input: string, ctx: CommandContext): boolean {
  const trimmed = input.trim()
  if (!trimmed.startsWith('/')) return false

  const parsed = parseSlashCommand(trimmed)
  if (!parsed) return false

  // Skills are ordinary agent messages. Menu selection only prepares the
  // draft; the completed skill-prefixed message is sent by the composer.
  if (getAvailableCommands().some((command) =>
    command.id === parsed.commandId && command.source === 'skill'
  )) return false

  // No built-in command currently accepts arguments. Treat argument-bearing
  // slash text as an ordinary message instead of triggering a destructive
  // action from pasted or conversational text.
  if (parsed.args.length > 0) return false

  switch (parsed.commandId) {
    case 'new':
      ctx.resetConversation()
      return true

    case 'resume':
      ctx.openRecentChats()
      return true

    case 'exit':
      ctx.exitApp()
      return true

    case 'bash':
      ctx.enterBashMode()
      return true

    case 'init':
      ctx.initProject()
      return true

    case 'settings':
      if (ctx.openModelMenu) {
        ctx.openModelMenu("models")
      } else {
        ctx.openSettings()
      }
      return true

    case 'usage':
      ctx.openUsage()
      return true

    case 'transcript':
      if (!ctx.toggleTranscript) return false
      ctx.toggleTranscript()
      return true

    case 'models':
    case 'catalog':
    case 'hardware':
    case 'cloud':
      if (!ctx.openModelMenu) return false
      ctx.openModelMenu(parsed.commandId)
      return true

    case 'autopilot':
      ctx.toggleAutopilot()
      return true

    default:
      ctx.showSystemMessage(`Unknown command: /${parsed.commandId}`)
      return true
  }
}
