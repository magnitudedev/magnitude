/**
 * Skill loader interface — abstracts the skill listing RPC so client-common
 * doesn't depend on the vanilla client. Each app provides its own impl
 * from its agent client.
 */
import type { SkillListEntry } from "@magnitudedev/sdk"

export interface SkillLoader {
  listSkills(cwd: string): Promise<readonly SkillListEntry[]>
}

export interface SlashCommandDefinition {
  id: string
  label: string
  description: string
  aliases?: string[]
  source?: 'skill'
  skillPath?: string
  featureFlag?: string
}

export async function loadSkillCommands(
  skillLoader: SkillLoader | null,
  cwd: string | null,
): Promise<SlashCommandDefinition[]> {
  if (!skillLoader || !cwd) return []
  const entries = await skillLoader.listSkills(cwd)
  return entries.map((s) => ({
    id: s.name,
    label: s.name,
    description: s.description,
    source: 'skill' as const,
    skillPath: s.path,
  }))
}

const SLASH_COMMANDS: SlashCommandDefinition[] = [
  { id: 'new',      label: 'new',      description: 'Start a new conversation' },
  { id: 'resume',   label: 'resume',   description: 'Resume a previous conversation' },
  { id: 'exit',     label: 'exit',     description: 'Exit Magnitude', aliases: ['quit', 'q'] },
  { id: 'bash',     label: 'bash',     description: 'Enter bash mode' },
  { id: 'init',     label: 'init',     description: 'Generate AGENTS.md for this project' },
  { id: 'settings',      label: 'settings',      description: 'Open settings', aliases: ['s'] },
  { id: 'usage',         label: 'usage',         description: 'View cloud subscription, limits, and recent usage', aliases: ['limits'] },
  { id: 'autopilot',     label: 'autopilot',     description: 'Toggle autopilot mode', featureFlag: 'MAGNITUDE_ENABLE_AUTOPILOT' },
]

let skillCommands: SlashCommandDefinition[] = []

export function registerSkillCommands(skills: SlashCommandDefinition[]) {
  skillCommands = skills
}

export function getAllCommands(): SlashCommandDefinition[] {
  return [...SLASH_COMMANDS, ...skillCommands]
}
