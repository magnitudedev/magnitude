import { getAllCommands, type SlashCommandDefinition } from './slash-commands'
import { logger } from '@magnitudedev/logger'

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
  /** Activate a skill by name and optional file path */
  activateSkill: (skillName: string, skillPath: string | undefined, args: string) => void
  /** Run the /init flow: explore codebase and generate AGENTS.md */
  initProject: () => void
  /** Handle /settings, /model, /provider, /skillset commands — open settings overlay */
  openSettings: (tab?: 'provider' | 'model' | 'skillset') => void
  /** Open the setup wizard */
  openSetup: () => void
  /** Open the browser setup overlay */
  openBrowserSetup: () => void
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

  for (const cmd of getAllCommands()) {
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
  if (!query) return [...getAllCommands()]

  const lowerQuery = query.toLowerCase()
  const prefixMatches: SlashCommandDefinition[] = []
  const substringMatches: SlashCommandDefinition[] = []

  for (const cmd of getAllCommands()) {
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

  logger.info({ command: parsed.commandId, args: parsed.args }, 'Slash command executed')

  // Check if this is a skill command
  const cmd = getAllCommands().find(c => c.id === parsed.commandId)
  if (cmd?.source === 'skill') {
    ctx.activateSkill(cmd.id, cmd.skillPath, parsed.args)
    return true
  }

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

    case 'setup':
      ctx.openSetup()
      return true

    case 'settings':
      ctx.openSettings()
      return true

    case 'model':
      ctx.openSettings('model')
      return true

    case 'provider':
      ctx.openSettings('provider')
      return true

    case 'skillset':
      ctx.openSettings('skillset')
      return true

    case 'browser-setup':
      ctx.openBrowserSetup()
      return true

    default:
      ctx.showSystemMessage(`Unknown command: /${parsed.commandId}`)
      return true
  }
}
