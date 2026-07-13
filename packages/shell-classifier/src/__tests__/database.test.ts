import { describe, expect, test } from 'bun:test'
import { isDatabaseForbidden, isDatabaseUtilityForbidden } from '../tools/database'

describe('database tool policy', () => {
  describe('isDatabaseForbidden', () => {
    for (const base of ['psql', 'mysql', 'mariadb', 'mongosh', 'mongo', 'redis-cli', 'sqlcmd']) {
      test(`${base} is forbidden`, () => {
        const reason = isDatabaseForbidden(base, [])
        expect(reason).toContain('Interactive DB shells')
      })
    }

    test('sqlite3 is allowed (local-only)', () => {
      expect(isDatabaseForbidden('sqlite3', [])).toBeNull()
    })

    test('non-db shell command returns null', () => {
      expect(isDatabaseForbidden('pg_dump', [])).toBeNull()
    })
  })

  describe('isDatabaseUtilityForbidden', () => {
    test('read-only export utilities are allowed', () => {
      expect(isDatabaseUtilityForbidden('pg_dump', ['mydb'])).toBeNull()
      expect(isDatabaseUtilityForbidden('mysqldump', ['app'])).toBeNull()
    })

    test('createdb is forbidden', () => {
      expect(isDatabaseUtilityForbidden('createdb', ['test_db'])).toBe('Creating databases mutates remote server state. Use read-only inspection commands instead.')
    })

    test('createuser is forbidden', () => {
      expect(isDatabaseUtilityForbidden('createuser', ['app_user'])).toBe('Creating database users mutates remote server state. Use read-only inspection commands instead.')
    })

    test('dropdb is forbidden', () => {
      expect(isDatabaseUtilityForbidden('dropdb', ['mydb'])).toContain('Dropping databases')
    })

    test('dropuser is forbidden', () => {
      expect(isDatabaseUtilityForbidden('dropuser', ['app_user'])).toContain('Dropping DB users')
    })

    test('pg_restore is always forbidden', () => {
      expect(isDatabaseUtilityForbidden('pg_restore', ['--clean', 'dump.sql'])).toBe('Restoring data mutates remote database state. Use `pg_dump` for read-only exports instead.')
      expect(isDatabaseUtilityForbidden('pg_restore', ['--if-exists', 'dump.sql'])).toBe('Restoring data mutates remote database state. Use `pg_dump` for read-only exports instead.')
      expect(isDatabaseUtilityForbidden('pg_restore', ['--create', 'dump.sql'])).toBe('Restoring data mutates remote database state. Use `pg_dump` for read-only exports instead.')
      expect(isDatabaseUtilityForbidden('pg_restore', ['dump.sql'])).toBe('Restoring data mutates remote database state. Use `pg_dump` for read-only exports instead.')
    })

    test('force + destructive token is forbidden', () => {
      expect(isDatabaseUtilityForbidden('dbutil', ['--force', 'delete', 'foo'])).toContain('Forced destructive DB utility operations')
      expect(isDatabaseUtilityForbidden('custom-db-tool', ['-f', '--action=drop', 'target'])).toContain('Forced destructive DB utility operations')
    })

    test('force without destructive token is allowed', () => {
      expect(isDatabaseUtilityForbidden('custom-db-tool', ['--force', 'list', 'users'])).toBeNull()
    })

    test('destructive token without force is allowed', () => {
      expect(isDatabaseUtilityForbidden('custom-db-tool', ['delete', 'target'])).toBeNull()
    })
  })
})