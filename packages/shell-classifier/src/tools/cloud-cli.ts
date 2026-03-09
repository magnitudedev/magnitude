import { cloudCommandPath, normalizeArgs } from '../util'

const AWS_GLOBAL_FLAGS_WITH_VALUE = new Set([
  '--profile',
  '--region',
  '--output',
  '--endpoint-url',
  '--cli-input-json',
  '--query',
])

const GCLOUD_GLOBAL_FLAGS_WITH_VALUE = new Set([
  '--project',
  '--account',
  '--configuration',
  '--impersonate-service-account',
  '--billing-project',
  '--format',
])

const AZ_GLOBAL_FLAGS_WITH_VALUE = new Set([
  '--subscription',
  '--resource-group',
  '-g',
  '--output',
  '--query',
])

const AWS_MUTATING_PREFIXES = [
  'create-',
  'delete-',
  'update-',
  'modify-',
  'put-',
  'remove-',
  'terminate-',
  'stop-',
  'start-',
  'reboot-',
  'revoke-',
  'disable-',
  'deregister-',
  'attach-',
  'detach-',
  'run-',
  'schedule-',
]

const AWS_READ_PREFIXES = ['describe-', 'list-', 'get-']
const AWS_S3_MUTATING = new Set(['cp', 'mv', 'rm', 'rb', 'sync', 'mb'])

const GCLOUD_MUTATING_VERBS = new Set([
  'create',
  'delete',
  'update',
  'remove',
  'reset',
  'resize',
  'start',
  'stop',
  'deploy',
  'destroy',
  'import',
  'export',
  'add',
  'set',
])

const GCLOUD_READ_VERBS = new Set(['describe', 'list', 'get', 'show'])

const AZ_MUTATING_VERBS = new Set([
  'create',
  'delete',
  'update',
  'start',
  'stop',
  'restart',
  'deallocate',
  'purge',
  'import',
  'export',
  'move',
  'swap',
])

const AZ_READ_VERBS = new Set(['show', 'list'])

function startsWithAny(token: string, prefixes: readonly string[]): boolean {
  return prefixes.some(prefix => token.startsWith(prefix))
}

function hasFlagValue(args: readonly string[], flag: string, value: string): boolean {
  const want = value.toLowerCase()
  for (let i = 0; i < args.length; i++) {
    const token = args[i]
    if (token === flag && i + 1 < args.length && args[i + 1] === want) return true
    if (token.startsWith(`${flag}=`) && token.slice(flag.length + 1) === want) return true
  }
  return false
}

function findLastVerb(path: readonly string[], verbs: ReadonlySet<string>): string | null {
  for (let i = path.length - 1; i >= 0; i--) {
    const token = path[i]
    if (verbs.has(token)) return token
  }
  return null
}

export function isCloudCliForbidden(base: string, args: readonly string[]): string | null {
  if (base === 'aws') return isAwsForbidden(args)
  if (base === 'gcloud') return isGcloudForbidden(args)
  if (base === 'az') return isAzForbidden(args)
  return null
}

export function isAwsForbidden(args: readonly string[]): string | null {
  const norm = normalizeArgs(args)
  const path = cloudCommandPath(norm, AWS_GLOBAL_FLAGS_WITH_VALUE)

  if (norm.includes('--version') || path[0] === '--version') return null
  if (path.length < 1) return null

  const service = path[0]
  const action = path[1]

  if (service === 'configure') return null
  if (!action) return null

  if (service === 's3') {
    if (AWS_S3_MUTATING.has(action)) {
      return `AWS s3 action '${action}' mutates remote bucket state and is forbidden. Use 'aws s3 ls' to inspect data instead.`
    }
    return null
  }

  if (startsWithAny(action, AWS_READ_PREFIXES)) return null

  if (startsWithAny(action, AWS_MUTATING_PREFIXES)) {
    return `AWS action '${action}' is mutating and forbidden. Use describe-*, list-*, or get-* commands instead.`
  }

  return null
}

export function isGcloudForbidden(args: readonly string[]): string | null {
  const norm = normalizeArgs(args)
  const path = cloudCommandPath(norm, GCLOUD_GLOBAL_FLAGS_WITH_VALUE)

  if (path.length < 1) return null

  const top = path[0]
  if (top === 'version' || top === 'info') return null
  if (top === 'config') return null

  if (top === 'auth') {
    const authVerb = path[1]
    if (authVerb === 'list') return null
    return `gcloud auth action '${authVerb ?? 'unknown'}' is mutating and forbidden. Use 'gcloud auth list' for read-only inspection.`
  }

  const mutatingPrefixes = ['create-', 'delete-', 'update-', 'remove-', 'reset-', 'resize-', 'start-', 'stop-', 'deploy-', 'destroy-', 'import-', 'export-', 'add-', 'set-']
  const readPrefixes = ['describe-', 'list-', 'get-', 'show-']

  let mutatingToken: string | null = null
  let readToken: string | null = null

  for (let i = path.length - 1; i >= 0; i--) {
    const token = path[i]
    if (!mutatingToken && (GCLOUD_MUTATING_VERBS.has(token) || startsWithAny(token, mutatingPrefixes))) {
      mutatingToken = token
    }
    if (!readToken && (GCLOUD_READ_VERBS.has(token) || startsWithAny(token, readPrefixes))) {
      readToken = token
    }
  }

  if (mutatingToken) {
    return `gcloud verb '${mutatingToken}' is mutating and forbidden. Use describe/list/get/show commands instead.`
  }

  if (readToken) return null

  return null
}

export function isAzForbidden(args: readonly string[]): string | null {
  const norm = normalizeArgs(args)
  const path = cloudCommandPath(norm, AZ_GLOBAL_FLAGS_WITH_VALUE)

  if (path.length < 1) return null

  const top = path[0]
  if (top === 'version' || top === 'configure' || top === 'init') return null
  if (top === 'config') return null

  const isDeploymentCreate =
    (path[0] === 'deployment' && path[2] === 'create') ||
    (path[0] === 'deployment' && path[1] === 'create')
  if (isDeploymentCreate && hasFlagValue(norm, '--mode', 'complete')) {
    return "Azure deployment 'create --mode complete' can delete unmanaged resources and is forbidden. Use show/list or what-if/incremental checks instead."
  }

  const verb = findLastVerb(path, new Set([...AZ_MUTATING_VERBS, ...AZ_READ_VERBS])) ?? path[path.length - 1]

  if (AZ_READ_VERBS.has(verb)) return null

  if (AZ_MUTATING_VERBS.has(verb)) {
    return `az verb '${verb}' is mutating and forbidden. Use show/list commands instead.`
  }

  return null
}