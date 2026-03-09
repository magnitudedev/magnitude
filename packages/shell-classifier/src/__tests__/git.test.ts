import { test, expect, describe } from 'bun:test'
import { isGitReadOnly } from '../tools/git'

/**
 * Adversarial test suite for git command classification.
 *
 * The rule is simple: only explicitly allowlisted read-only subcommands pass.
 * Everything else must be rejected. Tests are organized to try to sneak
 * write operations through via various git features.
 */

// Helper: split "git <args>" into args array (what the parser produces)
function args(command: string): string[] {
  // Strip leading "git " if present for convenience
  const s = command.startsWith('git ') ? command.slice(4) : command
  return s.split(/\s+/).filter(Boolean)
}

describe('git classification', () => {

  describe('allowed read-only commands', () => {
    const allowed = [
      'status',
      'status -s',
      'status --short',
      'status --porcelain',
      'log',
      'log --oneline',
      'log --oneline -20',
      'log --graph --all',
      'log --format="%H %s"',
      'diff',
      'diff HEAD',
      'diff HEAD~1',
      'diff --stat',
      'diff --cached',
      'diff --staged',
      'diff --name-only',
      'show',
      'show HEAD',
      'show HEAD:file.txt',
      'show --stat',
      'branch --list',
      'branch -l',
      'branch -a',
      'branch --all',
      'branch -r',
      'branch --remotes',
      'branch -v',
      'branch -vv',
      'branch --verbose',
      'branch --show-current',
      'branch --format=%(refname:short)',
      'branch -a -v',
      'branch --all --verbose',
    ]

    for (const cmd of allowed) {
      test(`git ${cmd} → readonly`, () => {
        expect(isGitReadOnly(args(cmd))).toBe(true)
      })
    }
  })

  describe('bare git branch (no args)', () => {
    test('git branch → readonly (lists branches)', () => {
      expect(isGitReadOnly(args('branch'))).toBe(true)
    })
  })

  describe('obvious write subcommands', () => {
    const forbidden = [
      'commit -m "msg"',
      'push',
      'push origin main',
      'push --force',
      'push --force-with-lease',
      'pull',
      'pull --rebase',
      'fetch',
      'merge main',
      'merge --no-ff feature',
      'rebase main',
      'rebase -i HEAD~3',
      'rebase --onto main feature',
      'reset HEAD~1',
      'reset --hard',
      'reset --hard HEAD~1',
      'reset --soft HEAD~1',
      'reset --mixed HEAD~1',
      'checkout main',
      'checkout -b new-branch',
      'checkout -- file.txt',
      'switch main',
      'switch -c new-branch',
      'restore file.txt',
      'restore --staged file.txt',
      'add .',
      'add -A',
      'add file.txt',
      'add -p',
      'rm file.txt',
      'rm --cached file.txt',
      'mv old.txt new.txt',
      'stash',
      'stash pop',
      'stash drop',
      'stash apply',
      'clean -f',
      'clean -fd',
      'clean -fx',
      'tag v1.0',
      'tag -a v1.0 -m "release"',
      'tag -d v1.0',
      'cherry-pick abc123',
      'revert HEAD',
      'revert abc123',
      'am patch.mbox',
      'apply patch.diff',
      'bisect start',
      'bisect bad',
      'bisect good abc123',
      'format-patch HEAD~3',
      'init',
      'init new-repo',
      'clone https://github.com/foo/bar',
      'remote add origin https://github.com/foo/bar',
      'remote remove origin',
      'submodule add https://github.com/foo/bar',
      'submodule update --init',
      'worktree add ../branch-dir main',
      'gc',
      'gc --aggressive',
      'prune',
      'fsck',
      'reflog expire --all',
      'update-ref HEAD abc123',
      'replace abc123 def456',
      'notes add -m "note"',
    ]

    for (const cmd of forbidden) {
      test(`git ${cmd} → forbidden`, () => {
        expect(isGitReadOnly(args(cmd))).toBe(false)
      })
    }
  })

  describe('config override attacks', () => {
    test('git -c core.hooksPath=/tmp status', () => {
      expect(isGitReadOnly(args('-c core.hooksPath=/tmp status'))).toBe(false)
    })

    test('git -c alias.status=!rm -rf / status', () => {
      expect(isGitReadOnly(args('-c alias.status=!rm status'))).toBe(false)
    })

    test('git -ccore.hooksPath=/tmp status (inline)', () => {
      expect(isGitReadOnly(args('-ccore.hooksPath=/tmp status'))).toBe(false)
    })

    test('git --config-env=TOKEN=GH_TOKEN status', () => {
      expect(isGitReadOnly(args('--config-env=TOKEN=GH_TOKEN status'))).toBe(false)
    })

    test('git --config-env TOKEN=GH_TOKEN status', () => {
      expect(isGitReadOnly(args('--config-env TOKEN=GH_TOKEN status'))).toBe(false)
    })

    test('git -c user.name=x log', () => {
      expect(isGitReadOnly(args('-c user.name=x log'))).toBe(false)
    })
  })

  describe('global option bypass attempts', () => {
    test('git -C /other/repo status → still readonly (just changes dir)', () => {
      // -C changes working dir, but status is still read-only
      expect(isGitReadOnly(args('-C /other/repo status'))).toBe(true)
    })

    test('git --git-dir=/other/.git status → still readonly', () => {
      expect(isGitReadOnly(args('--git-dir=/other/.git status'))).toBe(true)
    })

    test('git --work-tree=/other status → still readonly', () => {
      expect(isGitReadOnly(args('--work-tree=/other status'))).toBe(true)
    })

    test('git -C/tmp status (inline) → still readonly', () => {
      expect(isGitReadOnly(args('-C/tmp status'))).toBe(true)
    })

    test('git --git-dir=/tmp/.git push → forbidden (push is not readonly)', () => {
      expect(isGitReadOnly(args('--git-dir=/tmp/.git push'))).toBe(false)
    })
  })

  describe('branch write attempts', () => {
    test('git branch new-branch → forbidden (creates branch)', () => {
      expect(isGitReadOnly(args('branch new-branch'))).toBe(false)
    })

    test('git branch -d old-branch → forbidden (deletes)', () => {
      expect(isGitReadOnly(args('branch -d old-branch'))).toBe(false)
    })

    test('git branch -D old-branch → forbidden (force deletes)', () => {
      expect(isGitReadOnly(args('branch -D old-branch'))).toBe(false)
    })

    test('git branch -m new-name → forbidden (renames)', () => {
      expect(isGitReadOnly(args('branch -m new-name'))).toBe(false)
    })

    test('git branch -M new-name → forbidden (force renames)', () => {
      expect(isGitReadOnly(args('branch -M new-name'))).toBe(false)
    })

    test('git branch --set-upstream-to=origin/main → forbidden', () => {
      expect(isGitReadOnly(args('branch --set-upstream-to=origin/main'))).toBe(false)
    })

    test('git branch -u origin/main → forbidden', () => {
      expect(isGitReadOnly(args('branch -u origin/main'))).toBe(false)
    })

    test('git branch --copy old new → forbidden', () => {
      expect(isGitReadOnly(args('branch --copy old new'))).toBe(false)
    })

    test('git branch --edit-description → forbidden', () => {
      expect(isGitReadOnly(args('branch --edit-description'))).toBe(false)
    })
  })

  describe('diff/log with dangerous flags', () => {
    test('git diff --output=/tmp/patch → forbidden (writes file)', () => {
      expect(isGitReadOnly(args('diff --output=/tmp/patch'))).toBe(false)
    })

    test('git diff --output=patch.diff → forbidden', () => {
      expect(isGitReadOnly(args('diff --output=patch.diff'))).toBe(false)
    })

    test('git log --exec=cmd → forbidden', () => {
      expect(isGitReadOnly(args('log --exec=cmd'))).toBe(false)
    })

    test('git diff --ext-diff → forbidden (runs external program)', () => {
      expect(isGitReadOnly(args('diff --ext-diff'))).toBe(false)
    })

    test('git log --exec → forbidden', () => {
      expect(isGitReadOnly(args('log --exec'))).toBe(false)
    })
  })

  describe('subcommand smuggling attempts', () => {
    test('bare git (no subcommand) → forbidden', () => {
      expect(isGitReadOnly([])).toBe(false)
    })

    test('git --version → forbidden (not allowlisted)', () => {
      // --version is harmless but not on our allowlist — we're strict
      expect(isGitReadOnly(args('--version'))).toBe(false)
    })

    test('git help → forbidden (not allowlisted)', () => {
      expect(isGitReadOnly(args('help'))).toBe(false)
    })

    test('git config user.name → forbidden', () => {
      expect(isGitReadOnly(args('config user.name'))).toBe(false)
    })

    test('git config --get user.name → forbidden', () => {
      expect(isGitReadOnly(args('config --get user.name'))).toBe(false)
    })

    test('git archive HEAD → forbidden', () => {
      expect(isGitReadOnly(args('archive HEAD'))).toBe(false)
    })

    test('git bundle create repo.bundle HEAD → forbidden', () => {
      expect(isGitReadOnly(args('bundle create repo.bundle HEAD'))).toBe(false)
    })

    test('git fast-export HEAD → forbidden', () => {
      expect(isGitReadOnly(args('fast-export HEAD'))).toBe(false)
    })

    test('git fast-import → forbidden', () => {
      expect(isGitReadOnly(args('fast-import'))).toBe(false)
    })

    test('git filter-branch → forbidden', () => {
      expect(isGitReadOnly(args('filter-branch'))).toBe(false)
    })

    test('git shortlog → forbidden (not allowlisted)', () => {
      expect(isGitReadOnly(args('shortlog'))).toBe(false)
    })

    test('git describe → forbidden (not allowlisted)', () => {
      expect(isGitReadOnly(args('describe'))).toBe(false)
    })

    test('git rev-parse HEAD → forbidden (not allowlisted)', () => {
      expect(isGitReadOnly(args('rev-parse HEAD'))).toBe(false)
    })

    test('git ls-files → forbidden (not allowlisted)', () => {
      expect(isGitReadOnly(args('ls-files'))).toBe(false)
    })

    test('git cat-file -p HEAD → forbidden (not allowlisted)', () => {
      expect(isGitReadOnly(args('cat-file -p HEAD'))).toBe(false)
    })
  })

  describe('plumbing command attempts', () => {
    const plumbing = [
      'hash-object -w file.txt',
      'update-index --add file.txt',
      'write-tree',
      'commit-tree abc123 -m "msg"',
      'read-tree HEAD',
      'symbolic-ref HEAD refs/heads/main',
      'update-ref refs/heads/main abc123',
      'pack-refs --all',
      'mktag',
      'mktree',
    ]

    for (const cmd of plumbing) {
      test(`git ${cmd} → forbidden`, () => {
        expect(isGitReadOnly(args(cmd))).toBe(false)
      })
    }
  })

  describe('hook and alias exploitation', () => {
    test('git -c core.hooksPath=/evil status → forbidden', () => {
      expect(isGitReadOnly(args('-c core.hooksPath=/evil status'))).toBe(false)
    })

    test('git -c alias.log=!malicious log → forbidden', () => {
      expect(isGitReadOnly(args('-c alias.log=!malicious log'))).toBe(false)
    })

    test('git -c credential.helper=!cmd status → forbidden', () => {
      expect(isGitReadOnly(args('-c credential.helper=!cmd status'))).toBe(false)
    })

    test('git -c core.pager=!cmd log → forbidden', () => {
      expect(isGitReadOnly(args('-c core.pager=!cmd log'))).toBe(false)
    })

    test('git -c diff.external=!cmd diff → forbidden', () => {
      expect(isGitReadOnly(args('-c diff.external=!cmd diff'))).toBe(false)
    })
  })
})
