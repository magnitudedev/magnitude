/**
 * Git Command Classification
 *
 * Positive allowlist approach: only explicitly verified read-only git
 * subcommands are allowed. Everything else is forbidden.
 *
 * Allowed readonly: status, log, diff, show, branch (with readonly flags only).
 * Any git with -c config override is forbidden (can override hooks, aliases, etc).
 */

const READONLY_SUBCOMMANDS = ['status', 'log', 'diff', 'show', 'branch'] as const

/**
 * Returns true if the git args represent a read-only invocation.
 * Args should NOT include 'git' itself — just everything after it.
 */
export function isGitReadOnly(args: string[]): boolean {
  if (hasConfigOverride(args)) return false

  const sub = findSubcommand(args)
  if (!sub) return false

  switch (sub.command) {
    case 'status':
    case 'log':
    case 'diff':
    case 'show':
      return subcommandArgsReadOnly(sub.argsAfter)
    case 'branch':
      return subcommandArgsReadOnly(sub.argsAfter) && branchIsReadOnly(sub.argsAfter)
    default:
      return false
  }
}

// ─── Config override detection ──────────────────────────────

function hasConfigOverride(args: string[]): boolean {
  for (const arg of args) {
    // git -c key=value — standalone
    if (arg === '-c') return true
    // git -ckey=value — inline
    if (arg.startsWith('-c') && arg.length > 2) return true
    // git --config-env=KEY=ENVVAR
    if (arg === '--config-env' || arg.startsWith('--config-env=')) return true
  }
  return false
}

// ─── Subcommand extraction ──────────────────────────────────

interface SubcommandResult {
  command: string
  argsAfter: string[]
}

/**
 * Walk past git global options to find the subcommand.
 * Returns null if the first non-option word isn't in the readonly allowlist.
 */
function findSubcommand(args: string[]): SubcommandResult | null {
  let skipNext = false

  for (let i = 0; i < args.length; i++) {
    if (skipNext) {
      skipNext = false
      continue
    }

    const arg = args[i]

    // Global options with inline values: --git-dir=/foo, -C/tmp
    if (isGlobalOptionInline(arg)) continue

    // Global options that consume the next arg: -C /tmp, --git-dir /foo
    if (isGlobalOptionWithValue(arg)) {
      skipNext = true
      continue
    }

    // End-of-options marker or unknown flag — skip
    if (arg === '--' || arg.startsWith('-')) continue

    // First non-option word: must be an allowed subcommand
    if (READONLY_SUBCOMMANDS.includes(arg as typeof READONLY_SUBCOMMANDS[number])) {
      return { command: arg, argsAfter: args.slice(i + 1) }
    }
    return null
  }

  return null
}

// Git global options that take a separate value argument
const GLOBAL_OPTIONS_WITH_VALUE = new Set([
  '-C', '-c', '--config-env', '--exec-path',
  '--git-dir', '--namespace', '--super-prefix', '--work-tree',
])

function isGlobalOptionWithValue(arg: string): boolean {
  return GLOBAL_OPTIONS_WITH_VALUE.has(arg)
}

// Git global options with value packed inline: --git-dir=/foo, -C/tmp
function isGlobalOptionInline(arg: string): boolean {
  if (arg.startsWith('--config-env=')) return true
  if (arg.startsWith('--exec-path=')) return true
  if (arg.startsWith('--git-dir=')) return true
  if (arg.startsWith('--namespace=')) return true
  if (arg.startsWith('--super-prefix=')) return true
  if (arg.startsWith('--work-tree=')) return true
  // -C<path> (inline, not just -C)
  if (arg.startsWith('-C') && arg.length > 2) return true
  // -c<key=val> (inline) — but this is already caught by hasConfigOverride
  if (arg.startsWith('-c') && arg.length > 2) return true
  return false
}

// ─── Subcommand arg validation ──────────────────────────────

// Flags on subcommands that can trigger writes or arbitrary execution
const UNSAFE_SUBCOMMAND_FLAGS = new Set([
  '--output', '--ext-diff', '--textconv', '--exec', '--paginate',
])

function subcommandArgsReadOnly(args: string[]): boolean {
  for (const arg of args) {
    if (UNSAFE_SUBCOMMAND_FLAGS.has(arg)) return false
    if (arg.startsWith('--output=')) return false
    if (arg.startsWith('--exec=')) return false
  }
  return true
}

// ─── Branch-specific validation ─────────────────────────────

const BRANCH_READONLY_FLAGS = new Set([
  '--list', '-l', '--show-current',
  '-a', '--all', '-r', '--remotes',
  '-v', '-vv', '--verbose',
])

/**
 * `git branch` with no args is readonly (lists branches).
 * With args, every arg must be a known readonly flag.
 */
function branchIsReadOnly(args: string[]): boolean {
  if (args.length === 0) return true

  let sawReadOnlyFlag = false
  for (const arg of args) {
    if (BRANCH_READONLY_FLAGS.has(arg) || arg.startsWith('--format=')) {
      sawReadOnlyFlag = true
    } else {
      return false
    }
  }

  return sawReadOnlyFlag
}
