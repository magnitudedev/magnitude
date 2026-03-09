import { readFileSync, readdirSync, statSync } from 'fs'
import { join, relative } from 'path'

const FIXTURE_ROOT = join(__dirname, '../../../fixtures/behavior/mock-project')

function walkDir(dir: string, indent = ''): string[] {
  const entries = readdirSync(dir).sort()
  const lines: string[] = []
  for (const entry of entries) {
    if (entry === 'node_modules' || entry === '.git' || entry === 'dev.db') continue
    const fullPath = join(dir, entry)
    const isDir = statSync(fullPath).isDirectory()
    lines.push(`${indent}${entry}${isDir ? '/' : ''}`)
    if (isDir) {
      lines.push(...walkDir(fullPath, indent + '  '))
    }
  }
  return lines
}

function sessionContext(opts: { branch?: string; recentCommits?: string } = {}): string {
  const branch = opts.branch ?? 'main'
  const recentCommits = opts.recentCommits ?? 'a1b2c3d Initial scaffold'
  const folderStructure = walkDir(FIXTURE_ROOT).join('\n')

  return [
    '<session_context>',
    'Full name: Alex Chen',
    'Timezone: America/Los_Angeles',
    'Working directory: /Users/alex/task-manager',
    'Shell: zsh',
    'Platform: macos',
    `Git branch: ${branch}`,
    'Git status: (clean)',
    '',
    'Recent commits:',
    recentCommits,
    '',
    'Folder structure:',
    folderStructure,
    '</session_context>',
  ].join('\n')
}

export const mockProject = {
  read(path: string): string {
    return readFileSync(join(FIXTURE_ROOT, path), 'utf-8')
  },
  sessionContext,
}