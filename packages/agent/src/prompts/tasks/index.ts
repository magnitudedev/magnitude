import type { Skill } from '@magnitudedev/skills'

/**
 * Prompt text for task lifecycle hook reminders.
 * These are injected into the lead's context at specific moments.
 */

/**
 * Lightweight reference table for system prompt, driven by the active skills.
 */
export function renderTaskTypeReferenceTable(skills: Map<string, Skill>): string {
  const lines: string[] = []
  for (const [name, skill] of skills.entries()) {
    lines.push(`- **${skill.name}** (\`${name}\`) — ${skill.description}`)
  }
  return lines.join('\n')
}

/** Shown when a worker finishes and goes idle on a task */
export const taskIdleReminder = (agentId: string, taskId: string, title: string) =>
  `Worker ${agentId} for task ${taskId} ("${title}") has finished. Review output and either send feedback or mark complete. Re-consult the skill governing this task and evaluate the output against its quality bar before proceeding.`

/** Shown when a task is marked completed */
export const taskCompleteReminder = (taskId: string, title: string, skillPath: string) =>
  `Task ${taskId} ("${title}") completed. Consider whether the skill for this task should be updated based on what you learned. Skill file: ${skillPath}/SKILL.md`
