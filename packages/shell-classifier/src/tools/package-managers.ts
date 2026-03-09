const REASON_PUBLISH =
  '[Tool] publish/push mutates the remote package registry. Use local build/test/pack commands instead.'

function positionalArgs(args: readonly string[]): string[] {
  const out: string[] = []
  for (const raw of args) {
    const token = raw.toLowerCase()
    if (!token || token === '--') continue
    if (token.startsWith('-')) continue
    out.push(token)
  }
  return out
}

function matchesNested(pos: readonly string[], top: string, nested: readonly string[]): boolean {
  return pos[0] === top && pos.length >= 2 && nested.includes(pos[1]!)
}

function isNpmLikeForbidden(pos: readonly string[]): boolean {
  if (pos.length === 0) return false

  const first = pos[0]!
  if (['publish', 'unpublish', 'deprecate', 'adduser', 'login', 'star', 'unstar'].includes(first)) return true

  if (matchesNested(pos, 'dist-tag', ['add', 'rm'])) return true
  if (matchesNested(pos, 'owner', ['add', 'rm'])) return true
  if (matchesNested(pos, 'access', ['grant', 'revoke', 'public', 'restricted'])) return true
  if (matchesNested(pos, 'org', ['set', 'rm'])) return true
  if (matchesNested(pos, 'team', ['create', 'destroy', 'add', 'rm'])) return true
  if (matchesNested(pos, 'token', ['create', 'revoke'])) return true
  if (matchesNested(pos, 'hook', ['add', 'update', 'rm'])) return true

  return false
}

function hasAnyFlag(args: readonly string[], ...flags: readonly string[]): boolean {
  const set = new Set(flags.map((f) => f.toLowerCase()))
  for (const arg of args) {
    const token = arg.toLowerCase()
    if (set.has(token)) return true
    const eq = token.split('=')[0]
    if (eq && set.has(eq)) return true
  }
  return false
}

function cargoOrGemOwnerForbidden(pos: readonly string[], args: readonly string[]): boolean {
  if (pos[0] !== 'owner') return false
  return hasAnyFlag(args, '--add', '--remove')
}

export function isLangPackageManagerForbidden(base: string, args: readonly string[]): string | null {
  const b = base.toLowerCase()
  const pos = positionalArgs(args)

  if (b === 'npm' || b === 'pnpm') return isNpmLikeForbidden(pos) ? REASON_PUBLISH : null

  if (b === 'yarn') {
    if (pos[0] === 'publish' || pos[0] === 'login') return REASON_PUBLISH
    if (pos[0] === 'owner' && (pos[1] === 'add' || pos[1] === 'remove')) return REASON_PUBLISH
    if (pos[0] === 'tag' && (pos[1] === 'add' || pos[1] === 'remove')) return REASON_PUBLISH
    if (pos[0] === 'npm') {
      if (pos[1] === 'publish' || pos[1] === 'login') return REASON_PUBLISH
      if (pos[1] === 'tag' && (pos[2] === 'add' || pos[2] === 'remove')) return REASON_PUBLISH
      if (pos[1] === 'owner' && (pos[2] === 'add' || pos[2] === 'remove')) return REASON_PUBLISH
    }
    return null
  }

  if (b === 'bun') return pos[0] === 'publish' ? REASON_PUBLISH : null
  if (b === 'twine') return pos[0] === 'upload' ? REASON_PUBLISH : null
  if (b === 'poetry') return pos[0] === 'publish' ? REASON_PUBLISH : null
  if (b === 'uv') return pos[0] === 'publish' ? REASON_PUBLISH : null

  if (b === 'cargo') {
    if (pos[0] === 'publish' || pos[0] === 'yank') return REASON_PUBLISH
    if (cargoOrGemOwnerForbidden(pos, args)) return REASON_PUBLISH
    return null
  }

  if (b === 'gem') {
    if (pos[0] === 'push' || pos[0] === 'yank') return REASON_PUBLISH
    if (cargoOrGemOwnerForbidden(pos, args)) return REASON_PUBLISH
    return null
  }

  if (b === 'mvn') return pos[0] === 'deploy' ? REASON_PUBLISH : null

  if (b === 'gradle' || b === 'gradlew') {
    for (const token of pos) {
      if (token === 'publishtomavenlocal') continue
      if (token === 'publish' || token.startsWith('publish')) return REASON_PUBLISH
    }
    return null
  }

  if (b === 'dotnet') {
    if (pos[0] === 'nuget' && (pos[1] === 'push' || pos[1] === 'delete')) return REASON_PUBLISH
    return null
  }

  if (b === 'mix') {
    if (pos[0] === 'hex.publish' || pos[0] === 'hex.retire') return REASON_PUBLISH
    if (pos[0] === 'hex.owner' && (pos[1] === 'add' || pos[1] === 'remove' || pos[1] === 'transfer')) return REASON_PUBLISH
    return null
  }

  if (b === 'swift') {
    if (pos[0] === 'package-registry' && pos[1] === 'publish') return REASON_PUBLISH
    return null
  }

  return null
}