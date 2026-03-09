/**
 * Shell Command Classifier
 *
 * Tiers:
 * - readonly: positive allowlist of read-only commands
 * - normal: everything else that isn't forbidden
 * - forbidden: catastrophic patterns + non-allowlisted git commands
 */

import { resolve } from 'path'
import { parseShellCommand, type SimpleCommand } from './parser'
import { isGitReadOnly } from './tools/git'
import { isContainerForbidden } from './tools/container'
import { isKubectlForbidden, isHelmForbidden } from './tools/kubernetes'
import { isCloudCliForbidden } from './tools/cloud-cli'
import { isIacForbidden } from './tools/iac'
import { isDatabaseForbidden, isDatabaseUtilityForbidden } from './tools/database'
import {
  SYSADMIN_ALWAYS_FORBIDDEN,
  SYSADMIN_BLOCKLIST,
  PACKAGE_MANAGERS,
  isSysadminForbidden,
  getSysadminAlwaysForbiddenReason,
  isPackageManagerForbidden,
} from './tools/sysadmin'
import { isLangPackageManagerForbidden } from './tools/package-managers'
import type { ClassificationResult, ShellSafetyTier } from './types'

export function classifyShellCommand(command: string): ClassificationResult {
  const commands = parseShellCommand(command)
  let worst: ShellSafetyTier = 'readonly'
  let forbiddenReason: string | null = null

  for (const cmd of commands) {
    const result = classifyCommand(cmd)
    worst = worstTier(worst, result.tier)
    if (result.reason) forbiddenReason = result.reason
    if (worst === 'forbidden') return { tier: 'forbidden', reason: forbiddenReason }
  }

  return { tier: worst, reason: null }
}

export function isGitAllowed(command: string): boolean {
  const commands = parseShellCommand(command)

  for (const cmd of commands) {
    if (!cmd.name) continue
    const name = baseName(cmd.name)
    if (name !== 'git') continue
    if (!isGitReadOnly(cmd.args)) return false
  }

  return true
}

export function detectsOutsideCwd(command: string, cwd: string): boolean {
  const commands = parseShellCommand(command)

  for (const cmd of commands) {
    for (const redir of cmd.redirects) {
      if (isPathOutsideCwd(redir.target, cwd)) return true
    }

    if (cmd.name) {
      const name = baseName(cmd.name)
      if (WRITE_PATH_COMMANDS.has(name)) {
        for (const arg of cmd.args) {
          if (arg.startsWith('-')) continue
          if (isPathOutsideCwd(arg, cwd)) return true
        }
      }
    }
  }

  return false
}

// ─── Classification ─────────────────────────────────────────

const TIER_ORDER: Record<ShellSafetyTier, number> = {
  readonly: 0,
  normal: 1,
  forbidden: 2
}

function worstTier(a: ShellSafetyTier, b: ShellSafetyTier): ShellSafetyTier {
  return TIER_ORDER[a] >= TIER_ORDER[b] ? a : b
}

function classifyCommand(cmd: SimpleCommand): ClassificationResult {
  if (!cmd.name) {
    if (cmd.assignments.length === 0) return { tier: 'readonly', reason: null }
    let worst: ShellSafetyTier = 'readonly'
    let forbiddenReason: string | null = null
    for (const a of cmd.assignments) {
      const assignmentResult = classifyAssignmentValue(a.value)
      worst = worstTier(worst, assignmentResult.tier)
      if (assignmentResult.reason) forbiddenReason = assignmentResult.reason
      if (worst === 'forbidden') return { tier: 'forbidden', reason: forbiddenReason }
    }
    return { tier: worst, reason: null }
  }

  if (cmd.assignments.length > 0) return { tier: 'normal', reason: null }

  const hasRedirects = cmd.redirects.length > 0

  const name = baseName(cmd.name)
  const args = cmd.args

  if ((name === 'bash' || name === 'sh' || name === 'zsh') && isShellCInvocation(name, args)) {
    const innerCommand = extractShellCCommand(args)
    if (innerCommand) return classifyShellCommand(innerCommand)
    return { tier: 'normal', reason: null }
  }

  if (name === 'sudo') {
    const innerCmd: SimpleCommand = { assignments: [], name: args[0] ?? null, args: args.slice(1), redirects: cmd.redirects }
    const innerResult = classifyCommand(innerCmd)
    if (innerResult.tier === 'forbidden') return innerResult
    return { tier: 'normal', reason: null }
  }

  const forbiddenReason = isForbidden(name, args)
  if (forbiddenReason) return { tier: 'forbidden', reason: forbiddenReason }
  if (isReadOnly(name, args)) return { tier: hasRedirects ? 'normal' : 'readonly', reason: null }
  return { tier: 'normal', reason: null }
}

function classifyAssignmentValue(value: string): ClassificationResult {
  const subs = extractCommandSubstitutions(value)
  if (subs.length === 0) return { tier: 'readonly', reason: null }

  let worst: ShellSafetyTier = 'readonly'
  let forbiddenReason: string | null = null
  for (const sub of subs) {
    const result = classifyShellCommand(sub)
    worst = worstTier(worst, result.tier)
    if (result.reason) forbiddenReason = result.reason
    if (worst === 'forbidden') return { tier: 'forbidden', reason: forbiddenReason }
  }
  return { tier: worst, reason: null }
}

function extractCommandSubstitutions(value: string): string[] {
  const results: string[] = []
  let i = 0

  while (i < value.length) {
    if (value[i] === '$' && value[i + 1] === '(') {
      let depth = 1
      const start = i + 2
      let j = start
      while (j < value.length && depth > 0) {
        if (value[j] === '(') depth++
        else if (value[j] === ')') depth--
        if (depth > 0) j++
      }
      if (depth === 0) results.push(value.slice(start, j))
      i = j + 1
    } else {
      i++
    }
  }

  return results
}

function isShellCInvocation(_name: string, args: string[]): boolean {
  if (args.length < 2) return false
  if (args[0] === '-lc' || args[0] === '-c') return true
  if (args[0] === '-l' && args[1] === '-c') return true
  return false
}

function extractShellCCommand(args: string[]): string | null {
  if (args[0] === '-lc' || args[0] === '-c') return args[1] ?? null
  if (args[0] === '-l' && args[1] === '-c') return args[2] ?? null
  return null
}

const READONLY_COMMANDS = new Set([
  'cat', 'cd', 'cut', 'echo', 'expr', 'false', 'grep', 'head', 'id', 'ls',
  'nl', 'paste', 'pwd', 'rev', 'seq', 'stat', 'tail', 'tr', 'true', 'uname',
  'uniq', 'wc', 'which', 'whoami', 'tac', 'numfmt', 'less', 'more', 'file',
  'du', 'df', 'env', 'printenv', 'date', 'hostname', 'sort', 'dirname',
  'basename', 'realpath', 'readlink', 'test', '[', 'type', 'command',
  'jq', 'awk', 'gawk', 'mawk', 'nawk',
  'column', 'fmt', 'fold', 'comm', 'diff', 'strings', 'od', 'hexdump', 'tree',
])

function isReadOnly(cmd: string, args: string[]): boolean {
  if (READONLY_COMMANDS.has(cmd)) return true

  if (cmd === 'find') return isFindSafe(args)
  if (cmd === 'git') return isGitReadOnly(args)
  if (cmd === 'rg') return isRipgrepSafe(args)
  if (cmd === 'sed') return isSedSafe(args)
  if (cmd === 'base64') return isBase64Safe(args)
  if (cmd === 'yq') return isYqSafe(args)
  if (cmd === 'fd' || cmd === 'fdfind') return isFdSafe(args)
  if (cmd === 'ag') return true

  return false
}

function isFindSafe(args: string[]): boolean {
  const unsafeOptions = new Set(['-exec', '-execdir', '-ok', '-okdir', '-delete', '-fls', '-fprint', '-fprint0', '-fprintf'])
  return !args.some(arg => unsafeOptions.has(arg))
}

function isRipgrepSafe(args: string[]): boolean {
  const unsafeNoArg = new Set(['--search-zip', '-z'])
  const unsafeWithArg = ['--pre', '--hostname-bin']

  return !args.some(arg =>
    unsafeNoArg.has(arg) ||
    unsafeWithArg.some(opt => arg === opt || arg.startsWith(`${opt}=`))
  )
}

function isSedSafe(args: string[]): boolean {
  for (const arg of args) {
    if (!arg.startsWith('-')) continue
    if (arg === '--in-place' || arg.startsWith('--in-place=')) return false
    if (arg === '--file' || arg.startsWith('--file=')) return false
    if (arg.startsWith('--')) continue
    const flags = arg.slice(1)
    if (containsUnsafeSedFlag(flags)) return false
  }
  return true
}

function containsUnsafeSedFlag(flags: string): boolean {
  for (let i = 0; i < flags.length; i++) {
    const ch = flags[i]
    if (ch === 'i') return true
    if (ch === 'f') return true
    if (ch === 'e') return false
  }
  return false
}

function isBase64Safe(args: string[]): boolean {
  return !args.some(arg =>
    arg === '-o' || arg === '--output' ||
    arg.startsWith('--output=') ||
    (arg.startsWith('-o') && arg !== '-o')
  )
}

function isYqSafe(args: string[]): boolean {
  return !args.some(arg => arg === '-i' || arg === '--inplace' || arg.startsWith('--inplace='))
}

function isFdSafe(args: string[]): boolean {
  const unsafeOptions = new Set(['-x', '--exec', '-X', '--exec-batch'])
  return !args.some(arg => unsafeOptions.has(arg))
}

const SYSTEM_DIRS = new Set(['/etc', '/usr', '/System', '/bin', '/sbin', '/boot', '/var', '/lib', '/dev', '/proc', '/sys'])

const CONTAINER_TOOLS = new Set(['docker', 'podman', 'nerdctl'])
const CLOUD_CLIS = new Set(['aws', 'gcloud', 'az'])
const IAC_TOOLS = new Set(['terraform', 'terragrunt', 'pulumi', 'sst', 'cdk'])
const DB_SHELLS_FORBIDDEN = new Set(['psql', 'mysql', 'mariadb', 'mongosh', 'mongo', 'redis-cli', 'sqlcmd'])
const DB_UTILITY_TOOLS = new Set(['pg_dump', 'mysqldump', 'createdb', 'createuser', 'dropdb', 'dropuser', 'pg_restore'])
const LANG_PACKAGE_MANAGERS = new Set(['npm', 'pnpm', 'yarn', 'bun', 'twine', 'poetry', 'uv', 'cargo', 'gem', 'mvn', 'gradle', 'gradlew', 'dotnet', 'mix', 'swift'])

function isForbidden(cmd: string, args: string[]): string | null {
  if (cmd === ':') return 'This command is blocked as a shell-control sentinel, not a useful task action. Use a read-only check like `pwd` or `ls` instead.'
  if (cmd === 'mkfs') return 'Formatting filesystems can irreversibly erase disk data. Use read-only disk inspection like `lsblk` or `diskutil list` instead.'
  if (cmd === 'dd' && args.some(a => a.startsWith('if=') || a.startsWith('of=/dev'))) {
    return 'Raw device copy/write can destroy entire disks quickly. Use file-level copy commands on workspace files only.'
  }

  if (cmd === 'rm') {
    const hasForce = args.some(a => a === '-rf' || a === '-fr' || a === '-f')
    const targetsSystem = args.some(a => {
      if (a.startsWith('-')) return false
      return a === '/' || SYSTEM_DIRS.has(a) || Array.from(SYSTEM_DIRS).some(d => a.startsWith(d + '/'))
    })
    if (hasForce && targetsSystem) {
      return 'Force-deleting system paths can break the host environment irrecoverably. Delete only explicit project-local paths after listing them first.'
    }
  }

  if (cmd === 'git' && !isGitReadOnly(args)) {
    return 'Mutating git actions can permanently discard or rewrite history. Use read-only git commands like `git status`, `git log`, or `git diff`.'
  }

  return isForbiddenByToolPolicy(cmd, args)
}

function isForbiddenByToolPolicy(base: string, args: readonly string[]): string | null {
  if (CONTAINER_TOOLS.has(base)) return isContainerForbidden(base, args)
  if (base === 'kubectl') return isKubectlForbidden(args)
  if (base === 'helm') return isHelmForbidden(args)
  if (CLOUD_CLIS.has(base)) return isCloudCliForbidden(base, args)
  if (IAC_TOOLS.has(base)) return isIacForbidden(base, args)
  if (DB_SHELLS_FORBIDDEN.has(base)) return isDatabaseForbidden(base, args)
  if (DB_UTILITY_TOOLS.has(base)) return isDatabaseUtilityForbidden(base, args)
  if (SYSADMIN_ALWAYS_FORBIDDEN.has(base)) return getSysadminAlwaysForbiddenReason(base)
  if (SYSADMIN_BLOCKLIST.has(base)) return isSysadminForbidden(base, args)
  if (LANG_PACKAGE_MANAGERS.has(base)) {
    const result = isLangPackageManagerForbidden(base, args)
    if (result) return result
  }
  if (PACKAGE_MANAGERS.has(base)) return isPackageManagerForbidden(base, args)
  return null
}

const WRITE_PATH_COMMANDS = new Set(['rm', 'cp', 'mv', 'tee', 'mkdir', 'touch', 'chmod', 'chown', 'ln', 'install', 'rsync'])

function baseName(cmd: string): string {
  const i = cmd.lastIndexOf('/')
  return i === -1 ? cmd : cmd.slice(i + 1)
}

const ALLOWED_OUTSIDE_PREFIXES = ['/tmp/', '/dev/null']

export function isPathOutsideCwd(path: string, cwd: string): boolean {
  if (!path || path.startsWith('-')) return false

  const normalizedCwd = cwd.endsWith('/') ? cwd : cwd + '/'
  const cwdNoSlash = cwd.endsWith('/') ? cwd.slice(0, -1) : cwd

  let resolved: string
  if (path.startsWith('~')) {
    const home = process.env.HOME ?? process.env.USERPROFILE ?? ''
    resolved = resolve(home, path.slice(path.startsWith('~/') ? 2 : 1))
  } else {
    resolved = resolve(cwdNoSlash, path)
  }

  if (resolved === cwdNoSlash || resolved.startsWith(normalizedCwd)) return false

  if (ALLOWED_OUTSIDE_PREFIXES.some(p => resolved === p.slice(0, -1) || resolved.startsWith(p))) return false

  return true
}