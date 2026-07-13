const TERRAFORM_STATE_FORBIDDEN_SUBCOMMANDS = new Set(['rm', 'mv', 'push', 'replace-provider'])
const TERRAFORM_WORKSPACE_FORBIDDEN_SUBCOMMANDS = new Set(['new', 'delete', 'select'])
const PULUMI_STATE_FORBIDDEN_SUBCOMMANDS = new Set(['delete', 'unprotect'])
const PULUMI_STACK_FORBIDDEN_SUBCOMMANDS = new Set(['rm', 'init', 'import'])
const SST_STATE_READONLY_SUBCOMMANDS = new Set(['list', 'ls', 'show', 'get', 'inspect', 'help'])
const SST_STATE_FORBIDDEN_SUBCOMMANDS = new Set(['rm', 'remove', 'delete', 'mv', 'move', 'push', 'pull', 'replace-provider'])
const CDK_ALLOWED_TOP_LEVEL = new Set(['synth', 'diff', 'ls', 'list', 'doctor'])

const REASON_TERRAFORM_DESTROY = 'Destroy tears down managed infrastructure with high irreversible impact. Use `terraform plan`/`show` for safe review.'
const REASON_TERRAFORM_APPLY = 'Applying infrastructure changes mutates remote state and cloud resources. Use `terraform plan` for safe review.'
const REASON_TERRAFORM_IMPORT = 'Import mutates Terraform state and can alter ownership mapping. Use read-only state inspection first.'
const REASON_TERRAFORM_STATE_MUTATION = 'Direct state mutation can orphan resources and break future applies. Use `state list`/`state show` for read-only inspection.'
const REASON_TERRAFORM_FORCE_UNLOCK = 'Force unlock can corrupt concurrent state operations. Verify lock ownership and follow normal unlock process.'
const REASON_TERRAFORM_WORKSPACE_MUTATION = 'Workspace mutations can retarget or create state unexpectedly. Use `workspace list`/`workspace show` for read-only inspection.'
const REASON_TERRAFORM_TAINT = 'Taint/untaint mutates resource lifecycle behavior and influences apply outcomes. Use plan/show for diagnostics.'

const REASON_PULUMI_UP = 'Pulumi up deploys infrastructure changes and mutates remote state. Use `pulumi preview` to inspect changes safely.'
const REASON_PULUMI_DESTROY = 'Destroy removes stack resources and can cause downtime/data loss. Use `pulumi preview` to inspect changes safely.'
const REASON_PULUMI_STACK_MUTATION = 'Stack mutation commands can create/import/remove backend state. Use `pulumi stack ls/output` for read-only operations.'
const REASON_PULUMI_STATE_MUTATION = 'Direct state mutation can desynchronize state from real infrastructure. Prefer normal `pulumi up` workflows with preview.'
const REASON_PULUMI_CANCEL = 'Cancel mutates in-progress deployment state and can leave operations in an inconsistent status. Prefer read-only stack inspection.'

const REASON_SST_DEPLOY = 'Deploy mutates remote cloud infrastructure. Use `sst diff` for safe review.'
const REASON_SST_DEV = 'SST dev deploys infrastructure for dev mode and mutates remote state. Use `sst diff` for read-only comparison.'
const REASON_SST_REMOVE = 'Remove tears down SST-managed infrastructure and may cause outages. Use `sst diff` for safe inspection.'
const REASON_SST_STATE_MUTATION = 'State mutation can orphan resources and break future deployments. Use read-only state inspection commands only.'
const REASON_SST_SECRET_SET = 'Setting secrets mutates remote runtime configuration. Use read-only secret inspection workflows instead.'
const REASON_SST_SECRET_REMOVE = 'Removing secrets can break runtime config for deployed services. Use secret list/read or controlled rotation flows instead.'

const REASON_CDK_DEPLOY = 'CDK deploy mutates CloudFormation stacks in AWS. Use `cdk diff`/`cdk synth` for safe inspection.'
const REASON_CDK_DESTROY = 'CDK destroy tears down deployed infrastructure and can cause outages/data loss. Use `cdk diff` for read-only review.'
const REASON_CDK_BOOTSTRAP = 'CDK bootstrap provisions account-level resources and mutates remote infrastructure. Use read-only diagnostics instead.'
const REASON_CDK_WATCH = 'CDK watch can auto-deploy stack changes and mutate remote infrastructure. Use `cdk diff` for safe checks.'
const REASON_CDK_INIT = 'CDK init without `--generate-only` mutates environment state by installing dependencies. Use `cdk init --generate-only` for local scaffolding.'

export function isIacForbidden(base: string, args: readonly string[]): string | null {
  if (base === 'terraform' || base === 'terragrunt') return isTerraformFamilyForbidden(args)
  if (base === 'pulumi') return isPulumiForbidden(args)
  if (base === 'sst') return isSstForbidden(args)
  if (base === 'cdk') return isCdkForbidden(args)
  return null
}

function isTerraformFamilyForbidden(args: readonly string[]): string | null {
  const top = findFirstPositional(args)
  if (!top) return null

  if (top.token === 'destroy') return REASON_TERRAFORM_DESTROY
  if (top.token === 'apply') return REASON_TERRAFORM_APPLY
  if (top.token === 'import') return REASON_TERRAFORM_IMPORT
  if (top.token === 'force-unlock') return REASON_TERRAFORM_FORCE_UNLOCK
  if (top.token === 'taint' || top.token === 'untaint') return REASON_TERRAFORM_TAINT

  if (top.token === 'state') {
    const nested = findFirstPositional(args, top.index + 1)
    if (nested && TERRAFORM_STATE_FORBIDDEN_SUBCOMMANDS.has(nested.token)) return REASON_TERRAFORM_STATE_MUTATION
    return null
  }

  if (top.token === 'workspace') {
    const nested = findFirstPositional(args, top.index + 1)
    if (nested && TERRAFORM_WORKSPACE_FORBIDDEN_SUBCOMMANDS.has(nested.token)) return REASON_TERRAFORM_WORKSPACE_MUTATION
    return null
  }

  return null
}

function isPulumiForbidden(args: readonly string[]): string | null {
  const top = findFirstPositional(args)
  if (!top) return null

  if (top.token === 'up') return REASON_PULUMI_UP
  if (top.token === 'destroy') return REASON_PULUMI_DESTROY
  if (top.token === 'cancel') return REASON_PULUMI_CANCEL

  if (top.token === 'stack') {
    const nested = findFirstPositional(args, top.index + 1)
    if (nested && PULUMI_STACK_FORBIDDEN_SUBCOMMANDS.has(nested.token)) return REASON_PULUMI_STACK_MUTATION
    return null
  }

  if (top.token === 'state') {
    const nested = findFirstPositional(args, top.index + 1)
    if (nested && PULUMI_STATE_FORBIDDEN_SUBCOMMANDS.has(nested.token)) return REASON_PULUMI_STATE_MUTATION
    return null
  }

  return null
}

function isSstForbidden(args: readonly string[]): string | null {
  const top = findFirstPositional(args)
  if (!top) return null

  if (top.token === 'deploy') return REASON_SST_DEPLOY
  if (top.token === 'dev') return REASON_SST_DEV
  if (top.token === 'remove') return REASON_SST_REMOVE

  if (top.token === 'secret') {
    const nested = findFirstPositional(args, top.index + 1)
    if (nested?.token === 'set') return REASON_SST_SECRET_SET
    if (nested?.token === 'remove') return REASON_SST_SECRET_REMOVE
    return null
  }

  if (top.token === 'state') {
    const nested = findFirstPositional(args, top.index + 1)
    if (!nested) return REASON_SST_STATE_MUTATION
    if (SST_STATE_FORBIDDEN_SUBCOMMANDS.has(nested.token)) return REASON_SST_STATE_MUTATION
    if (!SST_STATE_READONLY_SUBCOMMANDS.has(nested.token)) return REASON_SST_STATE_MUTATION
    return null
  }

  return null
}

function isCdkForbidden(args: readonly string[]): string | null {
  const top = findFirstPositional(args)
  if (!top) return null

  if (CDK_ALLOWED_TOP_LEVEL.has(top.token)) return null
  if (top.token === 'deploy') return REASON_CDK_DEPLOY
  if (top.token === 'destroy') return REASON_CDK_DESTROY
  if (top.token === 'bootstrap') return REASON_CDK_BOOTSTRAP
  if (top.token === 'watch') return REASON_CDK_WATCH

  if (top.token === 'init') {
    if (hasFlag(args, '--generate-only')) return null
    return REASON_CDK_INIT
  }

  return null
}

function findFirstPositional(args: readonly string[], start = 0): { token: string, index: number } | null {
  for (let i = start; i < args.length; i++) {
    const token = args[i]
    if (!token || token === '--') continue
    if (token.startsWith('-')) {
      if (optionConsumesNextValue(token) && i + 1 < args.length) i++
      continue
    }
    return { token: token.toLowerCase(), index: i }
  }
  return null
}

function optionConsumesNextValue(token: string): boolean {
  if (token.includes('=')) return false
  return token === '--mode' || token === '--cwd' || token === '--stage' || token === '-chdir'
}

function hasFlag(args: readonly string[], ...flags: string[]): boolean {
  const normalizedFlags = new Set(flags.map((f) => f.toLowerCase()))
  for (const arg of args) {
    const token = arg.toLowerCase()
    if (normalizedFlags.has(token)) return true
    const eq = token.split('=')[0]
    if (normalizedFlags.has(eq)) return true
  }
  return false
}