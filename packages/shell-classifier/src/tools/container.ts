const FORBIDDEN_REMOTE_SUBCOMMANDS = new Map<string, string>([
  ['push', 'Pushing images mutates remote registries. Use `docker build` and `docker tag` for local image management.'],
  ['login', 'Docker login mutates remote authentication state. Use local image operations instead.'],
  ['logout', 'Docker logout mutates remote authentication state.'],
])

const PRUNE_SUBCOMMANDS = new Set(['system', 'container', 'image', 'volume', 'network', 'builder'])

const COMPOSE_DESTRUCTIVE_FLAGS = new Set(['-v', '--volumes', '--rmi', '--remove-orphans'])

const RUNLIKE_DIRECT = new Set(['run', 'create', 'exec'])
const RUNLIKE_COMPOSE = new Set(['up', 'run'])

const HOST_NAMESPACE_FLAGS = new Set([
  '--pid=host',
  '--ipc=host',
  '--uts=host',
  '--userns=host',
  '--network=host',
  '--net=host',
])

const CAP_RISK_FLAGS = new Set(['--cap-add=all', '--cap-add=sys_admin'])

const SECURITY_OPT_RISK_VALUES = new Set(['seccomp=unconfined', 'apparmor=unconfined'])

const SENSITIVE_MOUNT_PATH_PATTERNS = [
  '/var/run/docker.sock',
  '/etc',
  '/root',
  '/.ssh',
  '.ssh',
  '/.aws',
  '.aws',
  '/.config/gcloud',
  '.config/gcloud',
  '/.azure',
  '.azure',
]

const PRUNE_REASON = 'Prune can remove broad sets of images, volumes, and caches across workflows. Use targeted cleanup like `docker rm <id>` or `docker rmi <id>` instead.'
const COMPOSE_DOWN_REASON = 'This can delete persistent volumes/images and cause data loss. Use `docker compose stop` or `docker compose down` without destructive flags.'
const PRIVILEGED_REASON = 'Privileged mode removes key isolation and increases host impact risk. Run unprivileged and grant only minimal capabilities if required.'
const HOST_NAMESPACE_REASON = 'Host namespace sharing weakens containment and broadens blast radius. Use default namespaces and explicit port mappings instead.'
const CAP_SECURITY_REASON = 'Disabling sandbox controls or adding broad caps enables host-like behavior. Keep default security profiles and least-privilege capabilities.'
const SENSITIVE_MOUNT_REASON = 'Mounting sensitive host paths can expose secrets/system files to container code. Mount only required project directories.'

export function isContainerForbidden(_base: string, args: readonly string[]): string | null {
  if (args.length === 0) return null

  const directSubcommand = firstNonOptionToken(args)?.toLowerCase()
  if (directSubcommand != null) {
    const forbiddenReason = FORBIDDEN_REMOTE_SUBCOMMANDS.get(directSubcommand)
    if (forbiddenReason != null) return forbiddenReason
  }

  if (isPruneInvocation(args)) return PRUNE_REASON
  if (isComposeDownWithDestructiveFlags(args)) return COMPOSE_DOWN_REASON
  if (!isRunLikeInvocation(args)) return null

  if (hasPrivilegedFlag(args)) return PRIVILEGED_REASON
  if (hasHostNamespaceFlag(args)) return HOST_NAMESPACE_REASON
  if (hasCapRiskFlag(args) || hasUnconfinedSecurityOpt(args)) return CAP_SECURITY_REASON

  for (const spec of collectMountSpecs(args)) {
    if (isSensitiveMount(spec)) return SENSITIVE_MOUNT_REASON
  }

  return null
}

function isComposeSubcommand(args: readonly string[]): { isCompose: boolean; subcommand: string | null; rest: readonly string[] } {
  let i = 0
  while (i < args.length) {
    const token = args[i]
    if (isOptionToken(token)) {
      if (optionConsumesNext(token)) i += 2
      else i += 1
      continue
    }

    if (token !== 'compose') return { isCompose: false, subcommand: null, rest: [] }

    let j = i + 1
    while (j < args.length) {
      const t = args[j]
      if (isOptionToken(t)) {
        if (optionConsumesNext(t)) j += 2
        else j += 1
        continue
      }

      return { isCompose: true, subcommand: t, rest: args.slice(j + 1) }
    }

    return { isCompose: true, subcommand: null, rest: [] }
  }

  return { isCompose: false, subcommand: null, rest: [] }
}

function isPruneInvocation(args: readonly string[]): boolean {
  for (let i = 0; i < args.length - 1; i++) {
    if (PRUNE_SUBCOMMANDS.has(args[i]) && args[i + 1] === 'prune') return true
  }
  return false
}

function isComposeDownWithDestructiveFlags(args: readonly string[]): boolean {
  const compose = isComposeSubcommand(args)
  if (!compose.isCompose || compose.subcommand !== 'down') return false
  return args.some((arg) => COMPOSE_DESTRUCTIVE_FLAGS.has(arg))
}

function isRunLikeInvocation(args: readonly string[]): boolean {
  const compose = isComposeSubcommand(args)
  if (compose.isCompose) return compose.subcommand != null && RUNLIKE_COMPOSE.has(compose.subcommand)

  const first = firstNonOptionToken(args)
  return first != null && RUNLIKE_DIRECT.has(first)
}

function hasPrivilegedFlag(args: readonly string[]): boolean {
  return args.includes('--privileged')
}

function hasHostNamespaceFlag(args: readonly string[]): boolean {
  return args.some((arg) => HOST_NAMESPACE_FLAGS.has(arg.toLowerCase()))
}

function hasCapRiskFlag(args: readonly string[]): boolean {
  return args.some((arg) => CAP_RISK_FLAGS.has(arg.toLowerCase()))
}

function hasUnconfinedSecurityOpt(args: readonly string[]): boolean {
  for (let i = 0; i < args.length; i++) {
    const token = args[i]
    if (token === '--security-opt') {
      const value = args[i + 1]?.toLowerCase()
      if (value != null && SECURITY_OPT_RISK_VALUES.has(value)) return true
      continue
    }

    if (token.startsWith('--security-opt=')) {
      const value = token.slice('--security-opt='.length).toLowerCase()
      if (SECURITY_OPT_RISK_VALUES.has(value)) return true
    }
  }

  return false
}

function collectMountSpecs(args: readonly string[]): string[] {
  const specs: string[] = []

  for (let i = 0; i < args.length; i++) {
    const token = args[i]

    if (token === '-v' || token === '--volume' || token === '--mount') {
      const value = args[i + 1]
      if (value != null) specs.push(value)
      continue
    }

    if (token.startsWith('-v') && token.length > 2) {
      specs.push(token.slice(2))
      continue
    }

    if (token.startsWith('--volume=')) {
      specs.push(token.slice('--volume='.length))
      continue
    }

    if (token.startsWith('--mount=')) {
      specs.push(token.slice('--mount='.length))
      continue
    }
  }

  return specs
}

function isSensitiveMount(spec: string): boolean {
  const lower = spec.toLowerCase()

  if (lower.startsWith('/:')) return true

  const source = extractMountSource(lower)
  if (source === '/') return true

  return isSensitiveHostPath(source)
}

function extractMountSource(spec: string): string {
  if (spec.includes('type=') || spec.includes('source=') || spec.includes('src=')) {
    const parts = spec.split(',')
    for (const part of parts) {
      const trimmed = part.trim()
      if (trimmed.startsWith('source=')) return trimmed.slice('source='.length)
      if (trimmed.startsWith('src=')) return trimmed.slice('src='.length)
    }
    return spec
  }

  const idx = spec.indexOf(':')
  return idx === -1 ? spec : spec.slice(0, idx)
}

function isSensitiveHostPath(path: string): boolean {
  if (path === '/') return true

  for (const pattern of SENSITIVE_MOUNT_PATH_PATTERNS) {
    if (path.includes(pattern)) return true
  }

  return false
}

function firstNonOptionToken(args: readonly string[]): string | null {
  for (let i = 0; i < args.length; i++) {
    const token = args[i]
    if (isOptionToken(token)) {
      if (optionConsumesNext(token)) i += 1
      continue
    }
    return token
  }

  return null
}

function isOptionToken(token: string): boolean {
  return token.startsWith('-')
}

function optionConsumesNext(token: string): boolean {
  return token === '-f' || token === '--file' || token === '-p' || token === '--project-name' || token === '--profile' || token === '--context'
}