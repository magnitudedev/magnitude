/**
 * Session context builders.
 *
 * Builds the initial session context message and shared project context
 * from a SessionContext object.
 */

import type { SessionContext } from '../events'
import { getUserSkills } from '../skills'

/** Build the session context message content from a SessionContext object */
export function buildSessionContextContent(ctx: SessionContext): string {
  let content = '<session_context>\n'
  if (ctx.fullName) {
    content += 'Full name: ' + ctx.fullName + '\n'
  }
  content += 'Timezone: ' + ctx.timezone + '\n'
  content += buildProjectContext(ctx)
  if (ctx.git) {
    content += '\nRecent commits:\n' + ctx.git.recentCommits + '\n'
  }
  // const userSkills = getUserSkills(ctx.skills)
  // if (userSkills.length > 0) {
  //   content += '\n<available_skills>\nAdditional skills available in this project. Activate with skill(name).\n\n'
  //   content += userSkills.map(s => '- ' + s.name + ': ' + s.trigger).join('\n')
  //   content += '\n</available_skills>'
  // }

  content += '\n</session_context>'
  return content
}

/**
 * Shared project context for spawned agents.
 * Subset of session context relevant to code exploration.
 */
export function buildProjectContext(ctx: SessionContext): string {
  let content = ''
  content += 'Working directory: ' + ctx.cwd + '\n'
  content += 'Shell: ' + ctx.shell + '\n'
  content += 'Username: ' + ctx.username + '\n'
  content += 'Platform: ' + ctx.platform + '\n'
  if (ctx.git) {
    content += 'Git branch: ' + ctx.git.branch + '\n'
    content += 'Git status:\n' + (ctx.git.status || '(clean)') + '\n'
  }
  content += '\nFolder structure:\n' + ctx.folderStructure
  if (ctx.agentsFile) {
    content += '\n\n<agentfile filename="' + ctx.agentsFile.filename + '">\n' + ctx.agentsFile.content + '\n</agentfile>'
  }
  return content
}
