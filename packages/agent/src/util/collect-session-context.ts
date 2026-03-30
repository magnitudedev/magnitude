/**
 * Session Context Collection
 */

import { access, readFile } from 'fs/promises'
import { join } from 'path'
import type { StorageClient } from '@magnitudedev/storage'
import type { MagnitudeSlot } from '../model-slots'
import type { SessionContext, GitContext } from '../events'
import { scanSkills } from './skill-scanner'
import { runGitCommand } from './git-command'
import { knapsackFolderTree } from './folder-tree-knapsack'
import { truncateFolderTree } from './folder-tree-truncation'
import { buildTree } from './tree'
import { walk } from './walk'

// =============================================================================
// Constants
// =============================================================================

const FOLDER_TREE_BUDGET_TOKENS = 2500
const MAX_STATUS_LINES = 20

// =============================================================================
// Git Commands
// =============================================================================

function truncateStatus(status: string): string {
  const lines = status.split('\n')
  if (lines.length <= MAX_STATUS_LINES) {
    return status
  }
  const shown = lines.slice(0, MAX_STATUS_LINES)
  const remaining = lines.length - MAX_STATUS_LINES
  return [...shown, `... (${remaining} more files)`].join('\n')
}

async function collectGitContext(cwd: string): Promise<GitContext | null> {
  const [branch, status, commits, commitCount] = await Promise.all([
    runGitCommand(['branch', '--show-current'], cwd),
    runGitCommand(['status', '--short'], cwd),
    runGitCommand(['log', '--oneline', '-5'], cwd),
    runGitCommand(['rev-list', '--count', 'HEAD'], cwd)
  ])

  if (branch === null) {
    return null
  }

  let recentCommits = commits || '(no commits)'
  const totalCommits = commitCount ? parseInt(commitCount, 10) : 0
  if (totalCommits > 5 && commits) {
    recentCommits = `${commits}\n... (${totalCommits - 5} more)`
  }

  return {
    branch: branch || '(detached HEAD)',
    status: truncateStatus(status || ''),
    recentCommits
  }
}

// =============================================================================
// Folder Structure
// =============================================================================

async function collectFolderStructure(cwd: string): Promise<string> {
  try {
    await access(join(cwd, '.git'))
    const tree = await knapsackFolderTree(cwd, FOLDER_TREE_BUDGET_TOKENS)
    return tree || '(empty or no accessible folders)'
  } catch {
    const entries = await walk(cwd, cwd, 0, 2, null, {
      respectGitignore: true,
      collectSizes: true,
      collectMtimes: true,
    })
    const nodes = buildTree(entries)
    const tree = truncateFolderTree(nodes, FOLDER_TREE_BUDGET_TOKENS)
    return tree || '(empty or no accessible folders)'
  }
}

// =============================================================================
// Platform Normalization
// =============================================================================

function normalizePlatform(platform: NodeJS.Platform): 'macos' | 'linux' | 'windows' {
  switch (platform) {
    case 'darwin':
      return 'macos'
    case 'win32':
      return 'windows'
    default:
      return 'linux'
  }
}

// =============================================================================
// User Info
// =============================================================================

interface UserInfo {
  username: string
  fullName: string | null
}

async function runCommand(command: string, args: string[]): Promise<string | null> {
  try {
    const proc = Bun.spawn([command, ...args], {
      stdout: 'pipe',
      stderr: 'pipe'
    })
    const output = await new Response(proc.stdout).text()
    const exitCode = await proc.exited
    return exitCode === 0 ? output.trim() : null
  } catch {
    return null
  }
}

async function collectUserInfo(platform: 'macos' | 'linux' | 'windows'): Promise<UserInfo> {
  const username = process.env.USER || process.env.USERNAME || 'unknown'

  let fullName: string | null = null

  if (platform === 'macos') {
    fullName = await runCommand('id', ['-F'])
  } else if (platform === 'windows') {
    fullName = await runCommand('powershell', ['-Command', '(Get-LocalUser $env:USERNAME).FullName'])
  } else {
    const passwd = await runCommand('getent', ['passwd', username])
    if (passwd) {
      const gecos = passwd.split(':')[4]
      if (gecos) {
        fullName = gecos.split(',')[0] || null
      }
    }
  }

  if (fullName === '') fullName = null

  return { username, fullName }
}

// =============================================================================
// Agents File
// =============================================================================

async function readAgentsFile(cwd: string): Promise<{ filename: string; content: string } | null> {
  const filenames = ['AGENTS.md', 'CLAUDE.md']

  for (const filename of filenames) {
    try {
      const content = await readFile(join(cwd, filename), 'utf8')
      return { filename, content: content.trim() }
    } catch {
      continue
    }
  }

  return null
}

// =============================================================================
// Main Collection
// =============================================================================

export interface CollectSessionContextOptions {
  cwd?: string
  storage?: StorageClient<MagnitudeSlot>
  oneshot?: {
    prompt: string
  }
}

export async function collectSessionContext(opts?: CollectSessionContextOptions): Promise<Omit<SessionContext, 'workspacePath'>> {
  const cwd = opts?.cwd ?? process.cwd()
  const platform = normalizePlatform(process.platform)

  const [git, folderStructure, userInfo, agentsFile, skills] = await Promise.all([
    collectGitContext(cwd),
    collectFolderStructure(cwd),
    collectUserInfo(platform),
    readAgentsFile(cwd),
    scanSkills(cwd),
  ])

  return {
    cwd,
    platform,
    shell: process.env.SHELL?.split('/').pop() || 'bash',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    username: userInfo.username,
    fullName: userInfo.fullName,
    git,
    folderStructure,
    agentsFile,
    skills: skills.length > 0 ? skills : null,
    oneshot: opts?.oneshot,
  }
}
