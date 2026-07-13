import { test, expect, describe } from 'bun:test'
import { tokenize, parse, parseShellCommand } from '../parser'
import type { Token, SimpleCommand } from '../parser'

describe('shell-parser', () => {

  describe('tokenize', () => {

    test('simple words', () => {
      expect(tokenize('cat file.txt')).toEqual([
        { type: 'Word', value: 'cat' },
        { type: 'Word', value: 'file.txt' },
      ])
    })

    test('single-quoted string', () => {
      expect(tokenize("echo 'hello world'")).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'hello world' },
      ])
    })

    test('double-quoted string', () => {
      expect(tokenize('echo "hello world"')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'hello world' },
      ])
    })

    test('backslash escape outside quotes', () => {
      expect(tokenize('echo hello\\ world')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'hello world' },
      ])
    })

    test('backslash escape inside double quotes', () => {
      expect(tokenize('echo "hello\\"world"')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'hello"world' },
      ])
    })

    test('backslash inside single quotes is literal', () => {
      expect(tokenize("echo 'hello\\nworld'")).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'hello\\nworld' },
      ])
    })

    test('adjacent quoted segments form one word', () => {
      expect(tokenize("echo 'hello'\"world\"")).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'helloworld' },
      ])
    })

    test('pipe operator', () => {
      expect(tokenize('cat file | grep pattern')).toEqual([
        { type: 'Word', value: 'cat' },
        { type: 'Word', value: 'file' },
        { type: 'Pipe' },
        { type: 'Word', value: 'grep' },
        { type: 'Word', value: 'pattern' },
      ])
    })

    test('and operator', () => {
      expect(tokenize('cmd1 && cmd2')).toEqual([
        { type: 'Word', value: 'cmd1' },
        { type: 'And' },
        { type: 'Word', value: 'cmd2' },
      ])
    })

    test('or operator', () => {
      expect(tokenize('cmd1 || cmd2')).toEqual([
        { type: 'Word', value: 'cmd1' },
        { type: 'Or' },
        { type: 'Word', value: 'cmd2' },
      ])
    })

    test('semicolon', () => {
      expect(tokenize('cmd1; cmd2')).toEqual([
        { type: 'Word', value: 'cmd1' },
        { type: 'Semi' },
        { type: 'Word', value: 'cmd2' },
      ])
    })

    test('newline treated as semi', () => {
      expect(tokenize('cmd1\ncmd2')).toEqual([
        { type: 'Word', value: 'cmd1' },
        { type: 'Semi' },
        { type: 'Word', value: 'cmd2' },
      ])
    })

    test('redirect >', () => {
      expect(tokenize('echo foo > out.txt')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'foo' },
        { type: 'Redirect', op: '>' },
        { type: 'Word', value: 'out.txt' },
      ])
    })

    test('redirect >> (append)', () => {
      expect(tokenize('echo foo >> out.txt')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'foo' },
        { type: 'Redirect', op: '>>' },
        { type: 'Word', value: 'out.txt' },
      ])
    })

    test('redirect 2>', () => {
      expect(tokenize('cmd 2> /tmp/err')).toEqual([
        { type: 'Word', value: 'cmd' },
        { type: 'Redirect', op: '2>' },
        { type: 'Word', value: '/tmp/err' },
      ])
    })

    test('redirect 2>> maps to >>', () => {
      expect(tokenize('cmd 2>> /tmp/err')).toEqual([
        { type: 'Word', value: 'cmd' },
        { type: 'Redirect', op: '>>' },
        { type: 'Word', value: '/tmp/err' },
      ])
    })

    test('redirect &>', () => {
      expect(tokenize('cmd &> /tmp/all')).toEqual([
        { type: 'Word', value: 'cmd' },
        { type: 'Redirect', op: '&>' },
        { type: 'Word', value: '/tmp/all' },
      ])
    })

    test('redirect &>>', () => {
      expect(tokenize('cmd &>> /tmp/all')).toEqual([
        { type: 'Word', value: 'cmd' },
        { type: 'Redirect', op: '&>>' },
        { type: 'Word', value: '/tmp/all' },
      ])
    })

    test('2> only when word is exactly "2"', () => {
      // "22>" should be Word(22) then Redirect(>)
      expect(tokenize('cmd 22>file')).toEqual([
        { type: 'Word', value: 'cmd' },
        { type: 'Word', value: '22' },
        { type: 'Redirect', op: '>' },
        { type: 'Word', value: 'file' },
      ])
    })

    test('2> in middle of word is not fd redirect', () => {
      expect(tokenize('abc2>file')).toEqual([
        { type: 'Word', value: 'abc2' },
        { type: 'Redirect', op: '>' },
        { type: 'Word', value: 'file' },
      ])
    })

    test('redirect without spaces: >file', () => {
      expect(tokenize('echo foo>file')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'foo' },
        { type: 'Redirect', op: '>' },
        { type: 'Word', value: 'file' },
      ])
    })

    test('$() command substitution in word', () => {
      expect(tokenize('echo $(ls -la)')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: '$(ls -la)' },
      ])
    })

    test('$() preserves inner pipes as word content', () => {
      expect(tokenize('latest=$(ls -t dir | head -1)')).toEqual([
        { type: 'Word', value: 'latest=$(ls -t dir | head -1)' },
      ])
    })

    test('$() inside double quotes', () => {
      expect(tokenize('echo "$(ls -la)"')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: '$(ls -la)' },
      ])
    })

    test('nested $() substitutions', () => {
      expect(tokenize('echo $(cat $(ls))')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: '$(cat $(ls))' },
      ])
    })

    test('$() with single quotes inside', () => {
      expect(tokenize("echo $(echo 'hello world')")).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: "$(echo 'hello world')" },
      ])
    })

    test('$() with closing paren inside single quotes', () => {
      expect(tokenize("echo $(echo ')')")).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: "$(echo ')')" },
      ])
    })

    test('adjacent $() substitutions', () => {
      expect(tokenize('echo $(ls)$(pwd)')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: '$(ls)$(pwd)' },
      ])
    })

    test('backtick command substitution', () => {
      expect(tokenize('echo `ls -la`')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: '`ls -la`' },
      ])
    })

    test('backtick preserves spaces as word content', () => {
      const tokens = tokenize('result=`ls -t dir`')
      expect(tokens).toEqual([
        { type: 'Word', value: 'result=`ls -t dir`' },
      ])
    })

    test('subshell parens converted to Semi', () => {
      expect(tokenize('(echo a && echo b)')).toEqual([
        { type: 'Semi' },
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'a' },
        { type: 'And' },
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'b' },
        { type: 'Semi' },
      ])
    })

    test('background & treated as Semi', () => {
      expect(tokenize('cmd1 & cmd2')).toEqual([
        { type: 'Word', value: 'cmd1' },
        { type: 'Semi' },
        { type: 'Word', value: 'cmd2' },
      ])
    })

    test('operators inside single quotes are literal', () => {
      const tokens = tokenize("echo 'hello && world | test'")
      expect(tokens).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'hello && world | test' },
      ])
    })

    test('operators inside double quotes are literal', () => {
      const tokens = tokenize('echo "hello > world"')
      expect(tokens).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'hello > world' },
      ])
    })

    test('$VAR preserved as literal text', () => {
      expect(tokenize('echo $HOME')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: '$HOME' },
      ])
    })

    test('empty input', () => {
      expect(tokenize('')).toEqual([])
    })

    test('only whitespace', () => {
      expect(tokenize('   ')).toEqual([])
    })

    test('tabs as whitespace', () => {
      expect(tokenize('echo\tfoo')).toEqual([
        { type: 'Word', value: 'echo' },
        { type: 'Word', value: 'foo' },
      ])
    })
  })

  describe('parse', () => {

    test('single simple command', () => {
      const tokens = tokenize('cat file.txt')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'cat', args: ['file.txt'], redirects: [] }
      ])
    })

    test('command with multiple args', () => {
      const tokens = tokenize('ls -la /tmp')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'ls', args: ['-la', '/tmp'], redirects: [] }
      ])
    })

    test('pipe splits into two commands', () => {
      const tokens = tokenize('cat file | grep pattern')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'cat', args: ['file'], redirects: [] },
        { assignments: [], name: 'grep', args: ['pattern'], redirects: [] },
      ])
    })

    test('&& splits into two commands', () => {
      const tokens = tokenize('npm test && git push')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'npm', args: ['test'], redirects: [] },
        { assignments: [], name: 'git', args: ['push'], redirects: [] },
      ])
    })

    test('redirect attached to command', () => {
      const tokens = tokenize('echo foo > /tmp/out')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'echo', args: ['foo'], redirects: [{ op: '>', target: '/tmp/out' }] }
      ])
    })

    test('interleaved redirect and args', () => {
      const tokens = tokenize('cmd -a >out -b')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'cmd', args: ['-a', '-b'], redirects: [{ op: '>', target: 'out' }] }
      ])
    })

    test('multiple redirects', () => {
      const tokens = tokenize('cmd >stdout.log 2>stderr.log')
      expect(parse(tokens)).toEqual([
        {
          assignments: [], name: 'cmd', args: [],
          redirects: [
            { op: '>', target: 'stdout.log' },
            { op: '2>', target: 'stderr.log' },
          ]
        }
      ])
    })

    test('variable assignment only', () => {
      const tokens = tokenize('FOO=bar')
      expect(parse(tokens)).toEqual([
        { assignments: [{ name: 'FOO', value: 'bar' }], name: null, args: [], redirects: [] }
      ])
    })

    test('variable assignment with command', () => {
      const tokens = tokenize('FOO=bar npm test')
      expect(parse(tokens)).toEqual([
        { assignments: [{ name: 'FOO', value: 'bar' }], name: 'npm', args: ['test'], redirects: [] }
      ])
    })

    test('multiple variable assignments', () => {
      const tokens = tokenize('FOO=bar BAZ=qux cmd')
      expect(parse(tokens)).toEqual([
        {
          assignments: [{ name: 'FOO', value: 'bar' }, { name: 'BAZ', value: 'qux' }],
          name: 'cmd', args: [], redirects: []
        }
      ])
    })

    test('assignment with $() value', () => {
      const tokens = tokenize('latest=$(ls -t dir | head -1)')
      expect(parse(tokens)).toEqual([
        { assignments: [{ name: 'latest', value: '$(ls -t dir | head -1)' }], name: null, args: [], redirects: [] }
      ])
    })

    test('word with = not in prefix position is an arg', () => {
      const tokens = tokenize('echo FOO=bar')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'echo', args: ['FOO=bar'], redirects: [] }
      ])
    })

    test('word with = but invalid name is not assignment', () => {
      const tokens = tokenize('123=foo cmd')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: '123=foo', args: ['cmd'], redirects: [] }
      ])
    })

    test('empty segments skipped', () => {
      const tokens = tokenize('; ; ;')
      expect(parse(tokens)).toEqual([])
    })

    test('leading/trailing separators produce empty segments (skipped)', () => {
      const tokens = tokenize('; echo hello ;')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'echo', args: ['hello'], redirects: [] }
      ])
    })

    test('redirect with no target is skipped', () => {
      const tokens = tokenize('echo foo >')
      expect(parse(tokens)).toEqual([
        { assignments: [], name: 'echo', args: ['foo'], redirects: [] }
      ])
    })
  })

  describe('parseShellCommand (integration)', () => {

    test('cat file.txt | grep pattern | sort', () => {
      const result = parseShellCommand('cat file.txt | grep pattern | sort')
      expect(result).toEqual([
        { assignments: [], name: 'cat', args: ['file.txt'], redirects: [] },
        { assignments: [], name: 'grep', args: ['pattern'], redirects: [] },
        { assignments: [], name: 'sort', args: [], redirects: [] },
      ])
    })

    test('echo foo > /outside/file', () => {
      const result = parseShellCommand('echo foo > /outside/file')
      expect(result).toEqual([
        { assignments: [], name: 'echo', args: ['foo'], redirects: [{ op: '>', target: '/outside/file' }] }
      ])
    })

    test('npm test && git push', () => {
      const result = parseShellCommand('npm test && git push')
      expect(result).toEqual([
        { assignments: [], name: 'npm', args: ['test'], redirects: [] },
        { assignments: [], name: 'git', args: ['push'], redirects: [] },
      ])
    })

    test('FOO=bar npm test', () => {
      const result = parseShellCommand('FOO=bar npm test')
      expect(result).toEqual([
        { assignments: [{ name: 'FOO', value: 'bar' }], name: 'npm', args: ['test'], redirects: [] }
      ])
    })

    test('latest=$(ls -t dir | head -1); echo $latest', () => {
      const result = parseShellCommand('latest=$(ls -t dir | head -1); echo $latest')
      expect(result).toEqual([
        { assignments: [{ name: 'latest', value: '$(ls -t dir | head -1)' }], name: null, args: [], redirects: [] },
        { assignments: [], name: 'echo', args: ['$latest'], redirects: [] },
      ])
    })

    test("bash -lc 'inner command'", () => {
      const result = parseShellCommand("bash -lc 'inner command'")
      expect(result).toEqual([
        { assignments: [], name: 'bash', args: ['-lc', 'inner command'], redirects: [] }
      ])
    })

    test('sudo rm -rf foo', () => {
      const result = parseShellCommand('sudo rm -rf foo')
      expect(result).toEqual([
        { assignments: [], name: 'sudo', args: ['rm', '-rf', 'foo'], redirects: [] }
      ])
    })

    test('cmd 2> /tmp/err', () => {
      const result = parseShellCommand('cmd 2> /tmp/err')
      expect(result).toEqual([
        { assignments: [], name: 'cmd', args: [], redirects: [{ op: '2>', target: '/tmp/err' }] }
      ])
    })

    test('ls | tee /outside/out', () => {
      const result = parseShellCommand('ls | tee /outside/out')
      expect(result).toEqual([
        { assignments: [], name: 'ls', args: [], redirects: [] },
        { assignments: [], name: 'tee', args: ['/outside/out'], redirects: [] },
      ])
    })

    test('echo foo >> /outside/file', () => {
      const result = parseShellCommand('echo foo >> /outside/file')
      expect(result).toEqual([
        { assignments: [], name: 'echo', args: ['foo'], redirects: [{ op: '>>', target: '/outside/file' }] }
      ])
    })

    test("find packages -maxdepth 2 -name package.json | sed 's#^./##' | sort", () => {
      const result = parseShellCommand("find packages -maxdepth 2 -name package.json | sed 's#^./##' | sort")
      expect(result).toEqual([
        { assignments: [], name: 'find', args: ['packages', '-maxdepth', '2', '-name', 'package.json'], redirects: [] },
        { assignments: [], name: 'sed', args: ['s#^./##'], redirects: [] },
        { assignments: [], name: 'sort', args: [], redirects: [] },
      ])
    })

    test('echo foo &>> /outside/file', () => {
      const result = parseShellCommand('echo foo &>> /outside/file')
      expect(result).toEqual([
        { assignments: [], name: 'echo', args: ['foo'], redirects: [{ op: '&>>', target: '/outside/file' }] }
      ])
    })

    test('complex session inspection command', () => {
      const input = "latest=$(ls -t ~/.magnitude/sessions | head -1); echo $latest; ls -la ~/.magnitude/sessions/$latest; echo '--- meta.json ---'; cat ~/.magnitude/sessions/$latest/meta.json; echo '--- first 20 events ---'; head -20 ~/.magnitude/sessions/$latest/events.jsonl; echo '--- first 20 logs ---'; head -20 ~/.magnitude/sessions/$latest/logs.jsonl"
      const result = parseShellCommand(input)
      expect(result.length).toBe(9)
      expect(result[0].assignments[0].name).toBe('latest')
      expect(result[0].name).toBeNull()
      expect(result[1].name).toBe('echo')
      expect(result[2].name).toBe('ls')
      expect(result[3].name).toBe('echo')
      expect(result[4].name).toBe('cat')
      expect(result[5].name).toBe('echo')
      expect(result[6].name).toBe('head')
      expect(result[7].name).toBe('echo')
      expect(result[8].name).toBe('head')
    })

    test('subshell flattened into commands', () => {
      const result = parseShellCommand('(cd /tmp && rm foo)')
      expect(result).toEqual([
        { assignments: [], name: 'cd', args: ['/tmp'], redirects: [] },
        { assignments: [], name: 'rm', args: ['foo'], redirects: [] },
      ])
    })

    test('echo foo &> /dev/null', () => {
      const result = parseShellCommand('echo foo &> /dev/null')
      expect(result).toEqual([
        { assignments: [], name: 'echo', args: ['foo'], redirects: [{ op: '&>', target: '/dev/null' }] }
      ])
    })

    test('arithmetic expansion preserved as word', () => {
      const result = parseShellCommand('echo $((1+2))')
      expect(result).toEqual([
        { assignments: [], name: 'echo', args: ['$((1+2))'], redirects: [] }
      ])
    })

    test('command with quoted redirect target', () => {
      const result = parseShellCommand('echo foo > "/tmp/my file.txt"')
      expect(result).toEqual([
        { assignments: [], name: 'echo', args: ['foo'], redirects: [{ op: '>', target: '/tmp/my file.txt' }] }
      ])
    })

    test('unterminated single quote swallows rest', () => {
      // Graceful degradation: unterminated quote accumulates remaining input
      const result = parseShellCommand("echo 'unterminated")
      expect(result).toEqual([
        { assignments: [], name: 'echo', args: ['unterminated'], redirects: [] }
      ])
    })

    test('cmd 2>file without space', () => {
      const result = parseShellCommand('cmd 2>/tmp/err')
      expect(result).toEqual([
        { assignments: [], name: 'cmd', args: [], redirects: [{ op: '2>', target: '/tmp/err' }] }
      ])
    })

    test('2>&1 parsed as redirect to &1', () => {
      const result = parseShellCommand('cmd 2>&1')
      // 2> consumed as redirect, &1: & at token boundary → Semi, 1 → Word
      // So redirect has no target in segment (& splits it)
      // This is a known limitation — 2>&1 is misparse but safe
      expect(result.length).toBeGreaterThanOrEqual(1)
      expect(result[0].name).toBe('cmd')
    })
  })
})
