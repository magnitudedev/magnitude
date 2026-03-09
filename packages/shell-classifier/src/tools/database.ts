import { hasFlag } from '../util'

const DB_SHELLS_FORBIDDEN = new Set([
  'psql',
  'mysql',
  'mariadb',
  'mongosh',
  'mongo',
  'redis-cli',
  'sqlcmd',
])

const FORCE_FLAGS = ['--force', '-f']

const DESTRUCTIVE_VERB_TOKENS = new Set([
  'drop',
  'delete',
  'destroy',
  'purge',
  'truncate',
  'remove',
  'rm',
])

const REASON_DB_INTERACTIVE = 'Interactive DB shells allow arbitrary destructive queries the classifier cannot safely evaluate. Use read-only app-level diagnostics or exported artifacts instead.'
const REASON_DROPDB = 'Dropping databases is irreversible without verified backups. Use non-destructive metadata/listing commands instead.'
const REASON_DROPUSER = 'Dropping DB users can break authentication for running systems. Use user listing/inspection commands first.'
const REASON_CREATEDB = 'Creating databases mutates remote server state. Use read-only inspection commands instead.'
const REASON_CREATEUSER = 'Creating database users mutates remote server state. Use read-only inspection commands instead.'
const REASON_PG_RESTORE = 'Restoring data mutates remote database state. Use `pg_dump` for read-only exports instead.'
const REASON_FORCE_DESTRUCTIVE = 'Forced destructive DB utility operations bypass safeguards and confirmations. Use dry-run/list/inspect commands before changes.'

export function isDatabaseForbidden(base: string, _args: readonly string[]): string | null {
  if (DB_SHELLS_FORBIDDEN.has(base)) return REASON_DB_INTERACTIVE
  return null
}

export function isDatabaseUtilityForbidden(base: string, args: readonly string[]): string | null {
  if (base === 'dropdb') return REASON_DROPDB
  if (base === 'dropuser') return REASON_DROPUSER
  if (base === 'createdb') return REASON_CREATEDB
  if (base === 'createuser') return REASON_CREATEUSER
  if (base === 'pg_restore') return REASON_PG_RESTORE

  if (hasFlag(args, ...FORCE_FLAGS) && hasDestructiveToken(args)) {
    return REASON_FORCE_DESTRUCTIVE
  }

  return null
}


function hasDestructiveToken(args: readonly string[]): boolean {
  for (const arg of args) {
    const normalized = arg.toLowerCase()
    const fragments = normalized.split(/[=:/_-]+/).filter(Boolean)
    for (const fragment of fragments) {
      if (DESTRUCTIVE_VERB_TOKENS.has(fragment)) return true
    }
  }
  return false
}