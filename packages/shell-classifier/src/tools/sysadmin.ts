export const SYSADMIN_ALWAYS_FORBIDDEN = new Set([
  'shutdown', 'reboot', 'poweroff', 'halt',
  'fdisk', 'parted',
  'iptables', 'nft', 'ufw',
])

export const SYSADMIN_BLOCKLIST = new Set([
  'systemctl', 'service',
  'kill', 'pkill', 'killall',
  'mount', 'umount',
])

export const PACKAGE_MANAGERS = new Set([
  'apt', 'apt-get', 'yum', 'dnf', 'pacman', 'snap', 'brew',
])

const SYSTEMCTL_ALWAYS_FORBIDDEN_SUBCOMMANDS = new Set([
  'poweroff', 'reboot', 'halt', 'rescue', 'emergency', 'default',
])

const SYSTEMCTL_CRITICAL_ACTIONS = new Set(['stop', 'disable', 'mask'])
const CRITICAL_SERVICE_TARGETS = new Set(['network', 'networkmanager', 'sshd', 'docker'])
const PKILL_BROAD_NAMES = new Set(['node', 'python', 'java', 'sh'])
const PACKAGE_DESTRUCTIVE_TOKENS = new Set([
  'remove', 'purge', 'autoremove', 'dist-upgrade', 'full-upgrade', 'uninstall', 'cleanup',
])

const POWER_COMMANDS = new Set(['shutdown', 'reboot', 'poweroff', 'halt'])
const PARTITION_COMMANDS = new Set(['fdisk', 'parted'])
const FIREWALL_COMMANDS = new Set(['iptables', 'nft', 'ufw'])

const ALWAYS_FORBIDDEN_SUBCOMMAND_REASON =
  'Rescue/emergency/power service-management subcommands can destabilize system runtime. Use service status/log inspection instead.'

const CRITICAL_SERVICE_REASON =
  'Stopping or disabling critical services can cut access or break platform dependencies. Use status/log inspection without changing service state.'

const KILL_SYSTEM_PID_REASON =
  'Broad/system-init kills can crash the entire environment. Kill only verified task-specific PIDs.'

const PATTERN_HARD_KILL_REASON =
  'Pattern-based hard kills can terminate many unrelated processes. Use ps to identify exact PID and kill narrowly.'

const CRITICAL_MOUNT_REASON =
  'Mount changes on core paths can break OS/tooling immediately. Restrict mount actions to non-system temporary paths.'

const PACKAGE_GENERIC_REASON =
  'Destructive package operations can remove required tooling and destabilize the environment. Use install/list/search/info commands instead.'

const BREW_SERVICES_STOP_REASON =
  'Stopping services can disrupt active development dependencies. Use brew services list for inspection without state changes.'

const BREW_SERVICES_CLEANUP_REASON =
  'Cleanup can remove service artifacts and alter expected runtime behavior. Use targeted, reviewed service actions only.'

export function getSysadminAlwaysForbiddenReason(base: string): string {
  const b = base.toLowerCase()

  if (POWER_COMMANDS.has(b)) {
    return 'Host power-control commands can immediately terminate the working environment. Keep host lifecycle unchanged and continue with process-level diagnostics.'
  }

  if (PARTITION_COMMANDS.has(b)) {
    return 'Partition edits can irreversibly alter disks and destroy data. Use read-only disk layout inspection commands instead.'
  }

  if (FIREWALL_COMMANDS.has(b)) {
    return 'Firewall mutations can break connectivity and unrelated services. Use read-only network status inspection commands.'
  }

  return 'High-impact system administration commands can destabilize the environment. Use read-only diagnostics instead.'
}

export function isSysadminForbidden(base: string, args: readonly string[]): string | null {
  const b = base.toLowerCase()
  const lowerArgs = args.map(a => a.toLowerCase())

  if (b === 'systemctl' || b === 'service') {
    return systemServiceForbidden(b, lowerArgs)
  }

  if (b === 'kill') {
    return killForbidden(lowerArgs)
  }

  if (b === 'pkill' || b === 'killall') {
    return pkillKillallForbidden(lowerArgs)
  }

  if (b === 'mount' || b === 'umount') {
    for (const token of lowerArgs) {
      if (token.startsWith('-')) continue
      if (isCriticalMountPath(token)) return CRITICAL_MOUNT_REASON
    }
    return null
  }

  return null
}

export function isPackageManagerForbidden(base: string, args: readonly string[]): string | null {
  const b = base.toLowerCase()
  const lowerArgs = args.map(a => a.toLowerCase())
  const positionals = lowerArgs.filter(a => !a.startsWith('-'))

  if (b === 'brew') {
    if (positionals[0] === 'services') {
      const sub = positionals[1]
      if (sub === 'stop') return BREW_SERVICES_STOP_REASON
      if (sub === 'cleanup') return BREW_SERVICES_CLEANUP_REASON
    }

    if (positionals[0] === 'cleanup') return BREW_SERVICES_CLEANUP_REASON
  }

  for (const token of lowerArgs) {
    if (PACKAGE_DESTRUCTIVE_TOKENS.has(token)) return PACKAGE_GENERIC_REASON
  }

  return null
}

function systemServiceForbidden(base: string, args: readonly string[]): string | null {
  const positionals = args.filter(a => !a.startsWith('-'))

  if (base === 'systemctl') {
    const action = positionals[0]
    const target = normalizeServiceTarget(positionals[1])

    if (action && SYSTEMCTL_ALWAYS_FORBIDDEN_SUBCOMMANDS.has(action)) return ALWAYS_FORBIDDEN_SUBCOMMAND_REASON
    if (action && SYSTEMCTL_CRITICAL_ACTIONS.has(action) && target && CRITICAL_SERVICE_TARGETS.has(target)) return CRITICAL_SERVICE_REASON
    return null
  }

  // service: usually `service <target> <action>`, but tolerate swapped forms.
  const targetFirst = normalizeServiceTarget(positionals[0])
  const actionSecond = positionals[1]
  const actionFirst = positionals[0]
  const targetSecond = normalizeServiceTarget(positionals[1])

  if (actionFirst && SYSTEMCTL_ALWAYS_FORBIDDEN_SUBCOMMANDS.has(actionFirst)) return ALWAYS_FORBIDDEN_SUBCOMMAND_REASON
  if (actionSecond && SYSTEMCTL_ALWAYS_FORBIDDEN_SUBCOMMANDS.has(actionSecond)) return ALWAYS_FORBIDDEN_SUBCOMMAND_REASON

  if (actionSecond && SYSTEMCTL_CRITICAL_ACTIONS.has(actionSecond) && targetFirst && CRITICAL_SERVICE_TARGETS.has(targetFirst)) {
    return CRITICAL_SERVICE_REASON
  }

  if (actionFirst && SYSTEMCTL_CRITICAL_ACTIONS.has(actionFirst) && targetSecond && CRITICAL_SERVICE_TARGETS.has(targetSecond)) {
    return CRITICAL_SERVICE_REASON
  }

  return null
}

function normalizeServiceTarget(token?: string): string | null {
  if (!token) return null
  if (token.endsWith('.service')) return token.slice(0, -'.service'.length)
  return token
}

function killForbidden(args: readonly string[]): string | null {
  let expectSignalValue = false
  let afterDoubleDash = false

  for (let i = 0; i < args.length; i++) {
    const token = args[i]

    if (afterDoubleDash) {
      if (token === '1' || token === '-1') return KILL_SYSTEM_PID_REASON
      continue
    }

    if (token === '--') {
      afterDoubleDash = true
      continue
    }

    if (expectSignalValue) {
      expectSignalValue = false
      continue
    }

    if (token === '-s' || token === '--signal') {
      expectSignalValue = true
      continue
    }

    if (token === '-1') return KILL_SYSTEM_PID_REASON
    if (!token.startsWith('-') && token === '1') return KILL_SYSTEM_PID_REASON
  }

  return null
}

function pkillKillallForbidden(args: readonly string[]): string | null {
  const hasHardKill = hasHardKillFlag(args)
  if (!hasHardKill) return null

  for (const token of args) {
    if (token.startsWith('-')) continue
    if (PKILL_BROAD_NAMES.has(token)) return PATTERN_HARD_KILL_REASON
  }

  return null
}

function hasHardKillFlag(args: readonly string[]): boolean {
  for (let i = 0; i < args.length; i++) {
    const token = args[i]

    if (token === '-9') return true
    if (token === '-kill' || token === '-sigkill') return true
    if (token === '--signal=9' || token === '--signal=kill' || token === '--signal=sigkill') return true

    if ((token === '--signal' || token === '-s') && i + 1 < args.length) {
      const value = args[i + 1]
      if (value === '9' || value === 'kill' || value === 'sigkill') return true
    }
  }

  return false
}

function isCriticalMountPath(token: string): boolean {
  if (token === '/') return true
  if (token === '/system' || token.startsWith('/system/')) return true
  if (token === '/usr' || token.startsWith('/usr/')) return true
  if (token === '/etc' || token.startsWith('/etc/')) return true
  if (token === '/var' || token.startsWith('/var/')) return true
  return false
}