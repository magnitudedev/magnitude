const KUBECTL_FORBIDDEN = new Set([
  'apply',
  'create',
  'delete',
  'patch',
  'edit',
  'replace',
  'scale',
  'autoscale',
  'cordon',
  'uncordon',
  'drain',
  'taint',
  'run',
  'expose',
  'label',
  'annotate',
  'debug',
])

const HELM_FORBIDDEN = new Set(['install', 'upgrade', 'uninstall', 'rollback', 'test'])

const KUBECTL_REASON_MUTATION =
  'This kubectl command mutates cluster state and is forbidden. Use read-only alternatives like `kubectl get` or `kubectl describe`.'
const KUBECTL_REASON_ROLLOUT_MUTATION =
  'This kubectl rollout action mutates workloads and is forbidden. Use read-only rollout checks like `kubectl rollout status` or `kubectl rollout history`.'
const KUBECTL_REASON_SET_MUTATION =
  'This kubectl set command mutates resource configuration and is forbidden. Use read-only inspection like `kubectl get -o yaml` or `kubectl describe`.'
const KUBECTL_REASON_AUTH_RECONCILE =
  '`kubectl auth reconcile` mutates RBAC resources and is forbidden. Use read-only auth checks like `kubectl auth can-i` or `kubectl auth whoami`.'
const KUBECTL_REASON_CERT_MUTATION =
  'Certificate approval/denial mutates certificate state and is forbidden. Use read-only inspection like `kubectl get certificatesigningrequests`.'
const KUBECTL_REASON_FORCE =
  'Using `--force` can trigger destructive cluster mutations and is forbidden. Prefer read-only diagnostics like `kubectl get/describe`.'
const KUBECTL_REASON_GRACE_0 =
  'Using `--grace-period=0` can force abrupt workload termination and is forbidden. Prefer read-only diagnostics like `kubectl get/describe`.'
const KUBECTL_REASON_ALL =
  'Using `--all` can broaden destructive mutations and is forbidden. Prefer scoped read-only checks like `kubectl get` in a specific namespace.'
const KUBECTL_REASON_ALL_NAMESPACES =
  'Using `-A/--all-namespaces` can widen mutation blast radius and is forbidden. Prefer namespace-scoped read-only checks like `kubectl get/describe`.'

const HELM_REASON_MUTATION =
  'This helm command mutates release or cluster state and is forbidden. Use read-only alternatives like `helm list`, `helm status`, or `helm get`.'
const HELM_REASON_REGISTRY_MUTATION =
  'This helm registry command mutates registry auth or artifacts and is forbidden. Use read-only inspection like `helm search` or `helm show`.'
const HELM_REASON_REPO_MUTATION =
  'This helm repo command mutates repository configuration and is forbidden. Use read-only `helm repo list` and `helm search` instead.'
const HELM_REASON_PLUGIN_MUTATION =
  'This helm plugin command mutates local plugin state and is forbidden. Use built-in read-only helm commands instead.'
const HELM_REASON_FORCE =
  'Using `helm --force` can recreate resources disruptively and is forbidden. Prefer read-only checks like `helm status` or `helm get`.'

function normalize(args: readonly string[]): string[] {
  return args.map(a => a.toLowerCase())
}

function firstNonFlagIndex(args: readonly string[]): number {
  return args.findIndex(arg => !arg.startsWith('-'))
}

function nextNonFlagIndex(args: readonly string[], start: number): number {
  let i = Math.max(start, 0)
  while (i !== args.length) {
    if (!args[i].startsWith('-')) return i
    i += 1
  }
  return -1
}

function hasToken(args: readonly string[], token: string): boolean {
  return args.includes(token)
}

function hasGracePeriodZero(args: readonly string[]): boolean {
  let i = 0
  while (i !== args.length) {
    const arg = args[i]
    if (arg === '--grace-period=0') return true
    if (arg === '--grace-period' && args[i + 1] === '0') return true
    i += 1
  }
  return false
}

function hasForceFlag(args: readonly string[]): boolean {
  return hasToken(args, '--force')
}

export function isKubectlForbidden(args: readonly string[]): string | null {
  const normalized = normalize(args)
  if (normalized.length === 0) return null

  const cmd1Index = firstNonFlagIndex(normalized)
  if (cmd1Index === -1) return null
  const cmd1 = normalized[cmd1Index]

  const cmd2Index = nextNonFlagIndex(normalized, cmd1Index + 1)
  const cmd2 = cmd2Index === -1 ? null : normalized[cmd2Index]

  if (KUBECTL_FORBIDDEN.has(cmd1)) return KUBECTL_REASON_MUTATION

  if (cmd1 === 'rollout' && cmd2 && new Set(['restart', 'undo', 'pause', 'resume']).has(cmd2)) {
    return KUBECTL_REASON_ROLLOUT_MUTATION
  }

  if (cmd1 === 'set' && cmd2) return KUBECTL_REASON_SET_MUTATION

  if (cmd1 === 'auth' && cmd2 === 'reconcile') return KUBECTL_REASON_AUTH_RECONCILE

  if (cmd1 === 'certificate' && cmd2 && (cmd2 === 'approve' || cmd2 === 'deny')) {
    return KUBECTL_REASON_CERT_MUTATION
  }

  if (hasForceFlag(normalized)) return KUBECTL_REASON_FORCE
  if (hasGracePeriodZero(normalized)) return KUBECTL_REASON_GRACE_0
  if (hasToken(normalized, '--all')) return KUBECTL_REASON_ALL
  if (hasToken(normalized, '-a') || hasToken(normalized, '--all-namespaces')) return KUBECTL_REASON_ALL_NAMESPACES

  return null
}

export function isHelmForbidden(args: readonly string[]): string | null {
  const normalized = normalize(args)
  if (normalized.length === 0) return null

  const cmd1Index = firstNonFlagIndex(normalized)
  if (cmd1Index === -1) return null
  const cmd1 = normalized[cmd1Index]

  const cmd2Index = nextNonFlagIndex(normalized, cmd1Index + 1)
  const cmd2 = cmd2Index === -1 ? null : normalized[cmd2Index]

  if (HELM_FORBIDDEN.has(cmd1)) return HELM_REASON_MUTATION

  if (cmd1 === 'push') return HELM_REASON_REGISTRY_MUTATION

  if (cmd1 === 'registry' && cmd2 && (cmd2 === 'login' || cmd2 === 'logout')) {
    return HELM_REASON_REGISTRY_MUTATION
  }

  if (cmd1 === 'repo' && cmd2 && (cmd2 === 'add' || cmd2 === 'remove' || cmd2 === 'update')) {
    return HELM_REASON_REPO_MUTATION
  }

  if (cmd1 === 'plugin' && cmd2 && (cmd2 === 'install' || cmd2 === 'uninstall' || cmd2 === 'update')) {
    return HELM_REASON_PLUGIN_MUTATION
  }

  if (hasForceFlag(normalized)) return HELM_REASON_FORCE

  return null
}