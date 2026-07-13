import type { Skill } from '@magnitudedev/skills'

/**
 * Prompt text for task lifecycle hook reminders.
 * These are injected into the lead's context at specific moments.
 */

/**
 * Lightweight reference table for system prompt, driven by the active skills.
 */
export function renderSkillReferenceTable(skills: Map<string, Skill>): string {
  const lines: string[] = []
  for (const [name, skill] of skills.entries()) {
    lines.push(`- **${skill.name}** (\`${name}\`) — ${skill.description}`)
  }
  return lines.join('\n')
}

/** Shown when a worker finishes and goes idle on a task */
export const taskIdleReminder = (agentId: string, taskId: string, title: string) =>
  `Worker ${agentId} for task ${taskId} ("${title}") has finished. Review output and either send feedback or mark complete.`

/** Shown when a task is marked completed */
export const taskCompleteReminder = (taskId: string, title: string) =>
  `Task ${taskId} ("${title}") completed.`
