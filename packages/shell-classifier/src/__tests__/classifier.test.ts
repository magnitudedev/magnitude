import { test, expect, describe } from 'bun:test'
import { classifyShellCommand, isGitAllowed, isPathWithin, writesStayWithin } from '../classifier'

describe('shell-classifier', () => {

  describe('tier classification', () => {
    test('readonly commands classify as readonly', () => {
      expect(classifyShellCommand('cat file.txt').tier).toBe('readonly')
      expect(classifyShellCommand('ls -la').tier).toBe('readonly')
      expect(classifyShellCommand('grep pattern file').tier).toBe('readonly')
      expect(classifyShellCommand('git status').tier).toBe('readonly')
      expect(classifyShellCommand('git log --oneline').tier).toBe('readonly')
    })

    test('normal commands (unrecognized) classify as normal', () => {
      expect(classifyShellCommand('npm test').tier).toBe('normal')
      expect(classifyShellCommand('npm install').tier).toBe('normal')
      expect(classifyShellCommand('python script.py').tier).toBe('normal')
      expect(classifyShellCommand('make build').tier).toBe('normal')
    })

    test('formerly dangerous commands now classify as normal/forbidden', () => {
      expect(classifyShellCommand('rm -rf foo').tier).toBe('normal')
      expect(classifyShellCommand('git push --force').tier).toBe('forbidden')
      expect(classifyShellCommand('git reset --hard').tier).toBe('forbidden')
      expect(classifyShellCommand('git clean -f').tier).toBe('forbidden')
      expect(classifyShellCommand('kill -9 1234').tier).toBe('normal')
      expect(classifyShellCommand('killall node').tier).toBe('normal')
      expect(classifyShellCommand('chmod 777 file').tier).toBe('normal')
      expect(classifyShellCommand('chown root file').tier).toBe('normal')
    })

    test('forbidden commands classify as forbidden', () => {
      expect(classifyShellCommand('mkfs /dev/sda').tier).toBe('forbidden')
      expect(classifyShellCommand('rm -rf /usr').tier).toBe('forbidden')
      expect(classifyShellCommand('rm -rf /').tier).toBe('forbidden')
    })
  })

  describe('git forbidden cases', () => {
    for (const cmd of [
      'git commit -m "msg"',
      'git push',
      'git reset --hard',
      'git checkout main',
      'git merge main',
      'git stash',
      'git rebase main',
      'git add .',
      'git clean -f',
    ]) {
      test(`${cmd} is forbidden`, () => {
        expect(classifyShellCommand(cmd).tier).toBe('forbidden')
      })
    }
  })

  describe('sed', () => {
    test('sed -n Np is readonly', () => {
      expect(classifyShellCommand('sed -n 5p').tier).toBe('readonly')
      expect(classifyShellCommand('sed -n 1,10p').tier).toBe('readonly')
    })

    test('sed substitution is readonly', () => {
      expect(classifyShellCommand("sed 's/foo/bar/'").tier).toBe('readonly')
      expect(classifyShellCommand("sed 's#^./##'").tier).toBe('readonly')
    })

    test('sed -e expressions is readonly', () => {
      expect(classifyShellCommand("sed -e 's/foo/bar/' -e 's/baz/qux/'").tier).toBe('readonly')
    })

    test('sed -n with pattern is readonly', () => {
      expect(classifyShellCommand("sed -n '/pattern/p'").tier).toBe('readonly')
    })

    test('sed delete pattern is readonly', () => {
      expect(classifyShellCommand("sed '/^#/d'").tier).toBe('readonly')
    })

    test('sed reading a file without -i is readonly', () => {
      expect(classifyShellCommand("sed 's/foo/bar/' file.txt").tier).toBe('readonly')
    })

    test('sed -ne expression is readonly (-e consumes rest)', () => {
      expect(classifyShellCommand("sed -ne 's/foo/bar/p'").tier).toBe('readonly')
    })

    test('sed -nE is readonly', () => {
      expect(classifyShellCommand("sed -nE 's/foo/bar/'").tier).toBe('readonly')
    })

    test('sed -i is NOT readonly', () => {
      expect(classifyShellCommand("sed -i 's/foo/bar/' file.txt")).not.toBe('readonly')
    })

    test('sed -i.bak is NOT readonly', () => {
      expect(classifyShellCommand("sed -i.bak 's/foo/bar/' file.txt")).not.toBe('readonly')
    })

    test('sed --in-place is NOT readonly', () => {
      expect(classifyShellCommand("sed --in-place 's/foo/bar/' file.txt")).not.toBe('readonly')
      expect(classifyShellCommand("sed --in-place=.bak 's/foo/bar/' file.txt")).not.toBe('readonly')
    })

    test('sed -f is NOT readonly', () => {
      expect(classifyShellCommand("sed -f script.sed file.txt")).not.toBe('readonly')
    })

    test('sed combined flags with i is NOT readonly', () => {
      expect(classifyShellCommand("sed -ni 's/foo/bar/' file.txt")).not.toBe('readonly')
      expect(classifyShellCommand("sed -Ei 's/foo/bar/' file.txt")).not.toBe('readonly')
    })

    test('sed with redirect is escalated to normal', () => {
      expect(classifyShellCommand("sed 's/foo/bar/' file.txt > output.txt")).not.toBe('readonly')
    })
  })

  describe('awk', () => {
    test('awk is readonly', () => {
      expect(classifyShellCommand("awk '{print $1}'").tier).toBe('readonly')
      expect(classifyShellCommand("awk '{print $1}' file.txt").tier).toBe('readonly')
    })

    test('gawk is readonly', () => {
      expect(classifyShellCommand("gawk '{print $1}'").tier).toBe('readonly')
    })

    test('mawk is readonly', () => {
      expect(classifyShellCommand("mawk '{print $1}'").tier).toBe('readonly')
    })
  })

  describe('jq', () => {
    test('jq is readonly', () => {
      expect(classifyShellCommand("jq '.data[]'").tier).toBe('readonly')
      expect(classifyShellCommand("jq '.name' package.json").tier).toBe('readonly')
    })
  })

  describe('yq', () => {
    test('yq is readonly', () => {
      expect(classifyShellCommand("yq '.metadata.name' config.yaml").tier).toBe('readonly')
    })

    test('yq -i is NOT readonly', () => {
      expect(classifyShellCommand("yq -i '.version = \"2.0\"' config.yaml")).not.toBe('readonly')
    })

    test('yq --inplace is NOT readonly', () => {
      expect(classifyShellCommand("yq --inplace '.version = \"2.0\"' config.yaml")).not.toBe('readonly')
    })
  })

  describe('fd', () => {
    test('fd is readonly', () => {
      expect(classifyShellCommand("fd 'pattern'").tier).toBe('readonly')
      expect(classifyShellCommand('fd -e ts').tier).toBe('readonly')
    })

    test('fd --exec is NOT readonly', () => {
      expect(classifyShellCommand('fd -x rm')).not.toBe('readonly')
      expect(classifyShellCommand('fd --exec rm')).not.toBe('readonly')
      expect(classifyShellCommand('fd --exec-batch rm')).not.toBe('readonly')
    })
  })

  describe('ag', () => {
    test('ag is readonly', () => {
      expect(classifyShellCommand("ag 'pattern'").tier).toBe('readonly')
    })
  })

  describe('piped commands', () => {
    test('find | sed | sort is readonly', () => {
      expect(classifyShellCommand(
        "find packages -maxdepth 2 -name package.json | sed 's#^./##' | sort"
      ).tier).toBe('readonly')
    })

    test('cat | awk | sort is readonly', () => {
      expect(classifyShellCommand(
        "cat file.txt | awk '{print $2}' | sort -u"
      ).tier).toBe('readonly')
    })

    test('cat | jq is readonly', () => {
      expect(classifyShellCommand(
        "cat data.json | jq '.items[] | .name'"
      ).tier).toBe('readonly')
    })

    test('pipe with unsafe sed is NOT readonly', () => {
      expect(classifyShellCommand(
        "cat file.txt | sed -i 's/foo/bar/' file.txt"
      )).not.toBe('readonly')
    })
  })

  describe('readonly commands', () => {
    for (const cmd of ['tree', 'column', 'fmt', 'fold', 'comm', 'diff', 'strings', 'od', 'hexdump']) {
      test(`${cmd} is readonly`, () => {
        expect(classifyShellCommand(cmd).tier).toBe('readonly')
      })
    }
    for (const cmd of ['cat', 'ls', 'grep', 'head', 'tail', 'wc', 'sort', 'tr', 'echo', 'pwd']) {
      test(`${cmd} is readonly`, () => {
        expect(classifyShellCommand(cmd).tier).toBe('readonly')
      })
    }
  })

  describe('command substitution', () => {
    test('command with $() piped subshell is classified correctly', () => {
      expect(classifyShellCommand('latest=$(ls -t dir | head -1); echo $latest').tier).toBe('readonly')
    })

    test('full session inspection command is readonly', () => {
      expect(classifyShellCommand(
        "latest=$(ls -t ~/.magnitude/sessions | head -1); echo $latest; ls -la ~/.magnitude/sessions/$latest; echo '--- meta.json ---'; cat ~/.magnitude/sessions/$latest/meta.json; echo '--- first 20 events ---'; head -20 ~/.magnitude/sessions/$latest/events.jsonl; echo '--- first 20 logs ---'; head -20 ~/.magnitude/sessions/$latest/logs.jsonl"
      ).tier).toBe('readonly')
    })

    test('$() with unsafe inner command is not readonly', () => {
      expect(classifyShellCommand('result=$(npm test)').tier).toBe('normal')
    })

    test('$() with forbidden inner command is forbidden', () => {
      expect(classifyShellCommand('result=$(rm -rf /usr)').tier).toBe('forbidden')
    })

    test('composite command with forbidden git is forbidden', () => {
      expect(classifyShellCommand('echo ok && git push').tier).toBe('forbidden')
    })
  })

  describe('variable assignments', () => {
    test('bare assignment is readonly', () => {
      expect(classifyShellCommand('FOO=bar').tier).toBe('readonly')
    })

    test('env prefix form is normal', () => {
      expect(classifyShellCommand('FOO=bar npm test').tier).toBe('normal')
    })

    test('assignment with readonly $() is readonly', () => {
      expect(classifyShellCommand('latest=$(ls -t dir | head -1)').tier).toBe('readonly')
    })

    test('assignment with unsafe $() is normal', () => {
      expect(classifyShellCommand('result=$(npm install)').tier).toBe('normal')
    })
  })

  describe('writesStayWithin', () => {
    test('echo foo > /outside/file with cwd /project => false', () => {
      expect(writesStayWithin('echo foo > /outside/file', {}, '/project')).toBe(false)
    })

    test('echo foo >> /outside/file => false', () => {
      expect(writesStayWithin('echo foo >> /outside/file', {}, '/project')).toBe(false)
    })

    test('cmd 2> /tmp/err => true (tmp is allowlisted)', () => {
      expect(writesStayWithin('cmd 2> /tmp/err', {}, '/project')).toBe(true)
    })

    test('echo foo > ./inside/file with cwd /project => true', () => {
      expect(writesStayWithin('echo foo > ./inside/file', {}, '/project')).toBe(true)
    })

    test('ls | tee /outside/out => false', () => {
      expect(writesStayWithin('ls | tee /outside/out', {}, '/project')).toBe(false)
    })

    test('cat file && rm /etc/foo => false', () => {
      expect(writesStayWithin('cat file && rm /etc/foo', {}, '/project')).toBe(false)
    })

    test('rm ../outside with cwd /project/sub => false', () => {
      expect(writesStayWithin('rm ../outside', {}, '/project/sub')).toBe(false)
    })

    test('rm ./inside => true', () => {
      expect(writesStayWithin('rm ./inside', {}, '/project')).toBe(true)
    })

    test('npm install => true', () => {
      expect(writesStayWithin('npm install', {}, '/project')).toBe(true)
    })

    test('tee /tmp/out => true (tmp is allowlisted)', () => {
      expect(writesStayWithin('ls | tee /tmp/out', {}, '/project')).toBe(true)
    })

    test('echo foo > /dev/null => true (dev/null is allowlisted)', () => {
      expect(writesStayWithin('echo foo > /dev/null', {}, '/project')).toBe(true)
    })

    test('cp file /dev/sda => false (only /dev/null is allowlisted)', () => {
      expect(writesStayWithin('cp file /dev/sda', {}, '/project')).toBe(false)
    })

    test('workspace path is allowed when passed as additional root', () => {
      expect(writesStayWithin('echo foo > /Users/alice/.magnitude/sessions/123/workspace/note.txt', {}, '/project', '/Users/alice/.magnitude/sessions/123/workspace/')).toBe(true)
      expect(writesStayWithin('mkdir -p /Users/alice/.magnitude/sessions/123/workspace/tmp', {}, '/project', '/Users/alice/.magnitude/sessions/123/workspace')).toBe(true)
    })

    test('outside paths remain blocked even with workspace allowlist', () => {
      expect(writesStayWithin('echo foo > /Users/alice/.ssh/config', {}, '/project', '/Users/alice/.magnitude/sessions/123/workspace/')).toBe(false)
    })
  })

  describe('env var expansion in isPathWithin', () => {
    test('$HOME outside allowed roots is rejected', () => {
      expect(isPathWithin('$HOME/.bashrc', { HOME: '/Users/alice' }, '/project')).toBe(false)
    })

    test('$HOME inside allowed roots is allowed', () => {
      expect(isPathWithin('$HOME/sub', { HOME: '/project/sub' }, '/project')).toBe(true)
    })

    test('${VAR} syntax expanded', () => {
      expect(isPathWithin('${PROJECT_ROOT}/../secret', { PROJECT_ROOT: '/project' }, '/project')).toBe(false)
    })

    test('$M within workspace allowed', () => {
      expect(isPathWithin('$M/notes.md', { M: '/workspace' }, '/project', '/workspace')).toBe(true)
    })

    test('unknown var collapses to empty string', () => {
      expect(isPathWithin('foo/$NONEXISTENT/bar', {}, '/project')).toBe(true)
    })
  })

  describe('env var expansion in writesStayWithin', () => {
    test('redirect to $HOME outside roots rejected', () => {
      expect(writesStayWithin('echo x > $HOME/leak', { HOME: '/Users/alice' }, '/project')).toBe(false)
    })

    test('cp to $HOME rejected', () => {
      expect(writesStayWithin('cp file $HOME/.ssh/key', { HOME: '/Users/alice' }, '/project')).toBe(false)
    })

    test('redirect to $M within workspace allowed', () => {
      expect(writesStayWithin('echo x > $M/file', { M: '/workspace' }, '/project', '/workspace')).toBe(true)
    })
  })

  describe('isGitAllowed', () => {
    test('read-only git commands are allowed', () => {
      expect(isGitAllowed('git status')).toBe(true)
      expect(isGitAllowed('git log --oneline')).toBe(true)
      expect(isGitAllowed('git diff')).toBe(true)
      expect(isGitAllowed('git show HEAD')).toBe(true)
      expect(isGitAllowed('git branch --list')).toBe(true)
      expect(isGitAllowed('git branch -a')).toBe(true)
    })

    test('write git commands are not allowed', () => {
      expect(isGitAllowed('git push')).toBe(false)
      expect(isGitAllowed('git push --force')).toBe(false)
      expect(isGitAllowed('git commit -m "msg"')).toBe(false)
      expect(isGitAllowed('git add .')).toBe(false)
      expect(isGitAllowed('git checkout main')).toBe(false)
      expect(isGitAllowed('git reset --hard')).toBe(false)
      expect(isGitAllowed('git stash')).toBe(false)
      expect(isGitAllowed('git rebase main')).toBe(false)
    })

    test('non-git commands are allowed', () => {
      expect(isGitAllowed('npm test')).toBe(true)
      expect(isGitAllowed('ls -la')).toBe(true)
      expect(isGitAllowed('cat file.txt')).toBe(true)
    })

    test('mixed commands: non-git + allowed git', () => {
      expect(isGitAllowed('npm test && git log')).toBe(true)
      expect(isGitAllowed('npm test && git status')).toBe(true)
    })

    test('mixed commands: non-git + disallowed git', () => {
      expect(isGitAllowed('npm test && git push')).toBe(false)
      expect(isGitAllowed('git status && git commit -m "msg"')).toBe(false)
    })

    test('git with -c config override is not allowed', () => {
      expect(isGitAllowed('git -c user.name=x status')).toBe(false)
    })
  })
})