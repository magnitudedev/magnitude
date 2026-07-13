import { describe, expect, test } from 'bun:test'
import { isLangPackageManagerForbidden } from '../tools/package-managers'

describe('language package manager registry policy', () => {
  test('npm/pnpm forbidden commands', () => {
    expect(isLangPackageManagerForbidden('npm', ['publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['dist-tag', 'add', 'pkg@1', 'latest'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['owner', 'rm', 'user', 'pkg'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['access', 'grant', 'read-write', 'team', 'pkg'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['org', 'set', 'org:team', 'user'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['team', 'destroy', 'org:team'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['token', 'create'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['hook', 'update', 'id', 'url', 'secret'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['star', 'pkg'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('pnpm', ['unpublish', 'pkg@1.0.0'])).not.toBeNull()
  })

  test('npm/pnpm allowed commands', () => {
    expect(isLangPackageManagerForbidden('npm', ['install'])).toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['dist-tag', 'ls', 'pkg'])).toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['owner', 'ls', 'pkg'])).toBeNull()
    expect(isLangPackageManagerForbidden('npm', ['token', 'list'])).toBeNull()
    expect(isLangPackageManagerForbidden('pnpm', ['whoami'])).toBeNull()
  })

  test('yarn classic + berry rules', () => {
    expect(isLangPackageManagerForbidden('yarn', ['publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('yarn', ['login'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('yarn', ['owner', 'add', 'user', 'pkg'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('yarn', ['tag', 'remove', 'pkg', 'latest'])).not.toBeNull()

    expect(isLangPackageManagerForbidden('yarn', ['npm', 'publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('yarn', ['npm', 'login'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('yarn', ['npm', 'tag', 'add', 'pkg', 'latest'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('yarn', ['npm', 'owner', 'remove', 'u', 'pkg'])).not.toBeNull()

    expect(isLangPackageManagerForbidden('yarn', ['install'])).toBeNull()
    expect(isLangPackageManagerForbidden('yarn', ['add', 'react'])).toBeNull()
    expect(isLangPackageManagerForbidden('yarn', ['npm', 'info', 'react'])).toBeNull()
  })

  test('bun, twine, poetry, uv', () => {
    expect(isLangPackageManagerForbidden('bun', ['publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('bun', ['run', 'test'])).toBeNull()

    expect(isLangPackageManagerForbidden('twine', ['upload', 'dist/*'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('twine', ['check', 'dist/*'])).toBeNull()

    expect(isLangPackageManagerForbidden('poetry', ['publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('poetry', ['build'])).toBeNull()

    expect(isLangPackageManagerForbidden('uv', ['publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('uv', ['sync'])).toBeNull()
  })

  test('cargo and gem owner flag checks', () => {
    expect(isLangPackageManagerForbidden('cargo', ['publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('cargo', ['yank', 'crate@1.0.0'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('cargo', ['owner', '--add', 'alice', 'crate'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('cargo', ['owner', '--remove=bob', 'crate'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('cargo', ['build'])).toBeNull()
    expect(isLangPackageManagerForbidden('cargo', ['login'])).toBeNull()

    expect(isLangPackageManagerForbidden('gem', ['push', 'pkg.gem'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('gem', ['yank', 'pkg', '-v', '1.0.0'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('gem', ['owner', '--add', 'alice', 'pkg'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('gem', ['owner', '--remove', 'bob', 'pkg'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('gem', ['install', 'rake'])).toBeNull()
  })

  test('mvn/gradle/gradlew', () => {
    expect(isLangPackageManagerForbidden('mvn', ['deploy'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('mvn', ['test'])).toBeNull()

    expect(isLangPackageManagerForbidden('gradle', ['publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('gradle', ['publishToMavenCentral'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('gradle', ['publishToMavenLocal'])).toBeNull()
    expect(isLangPackageManagerForbidden('gradlew', ['build'])).toBeNull()
  })

  test('dotnet, mix, swift', () => {
    expect(isLangPackageManagerForbidden('dotnet', ['nuget', 'push', 'nupkg'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('dotnet', ['nuget', 'delete', 'pkg', '1.0.0'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('dotnet', ['publish'])).toBeNull()
    expect(isLangPackageManagerForbidden('dotnet', ['nuget', 'list', 'source'])).toBeNull()

    expect(isLangPackageManagerForbidden('mix', ['hex.publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('mix', ['hex.retire', 'pkg', '1.0.0'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('mix', ['hex.owner', 'transfer', 'pkg', 'org'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('mix', ['deps.get'])).toBeNull()

    expect(isLangPackageManagerForbidden('swift', ['package-registry', 'publish'])).not.toBeNull()
    expect(isLangPackageManagerForbidden('swift', ['build'])).toBeNull()
  })
})