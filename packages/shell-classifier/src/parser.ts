/**
 * Shell Parser
 *
 * State-machine tokenizer + single-pass parser for shell command strings.
 * No regex. Produces a flat list of SimpleCommands for security classification.
 *
 * Handles: single/double quoting, backslash escaping, pipes, &&, ||, ;,
 * redirects (>, >>, 2>, &>, &>>), command substitution $(), backtick
 * substitution, parenthesized subshells, variable assignments.
 */

// ─── Tokens ──────────────────────────────────────────────────

export type RedirectOp = '>' | '>>' | '2>' | '&>' | '&>>'

export type Token =
  | { type: 'Word'; value: string }
  | { type: 'Pipe' }
  | { type: 'And' }
  | { type: 'Or' }
  | { type: 'Semi' }
  | { type: 'Redirect'; op: RedirectOp }

// ─── AST ─────────────────────────────────────────────────────

export interface Assignment {
  name: string
  value: string
}

export interface Redirect {
  op: RedirectOp
  target: string
}

export interface SimpleCommand {
  assignments: Assignment[]
  name: string | null
  args: string[]
  redirects: Redirect[]
}

// ─── Tokenizer ───────────────────────────────────────────────

export function tokenize(input: string): Token[] {
  const tokens: Token[] = []
  const len = input.length
  let word = ''
  let i = 0

  // Top-level quoting state
  let singleQ = false
  let doubleQ = false
  let escaped = false

  // $() tracking (with inner quoting for paren balancing)
  let subDepth = 0
  let subSingleQ = false
  let subDoubleQ = false
  let subEscaped = false

  // Backtick tracking
  let backtick = false

  // Subshell (bare parens) depth
  let parenDepth = 0

  function flush() {
    if (word.length > 0) {
      tokens.push({ type: 'Word', value: word })
      word = ''
    }
  }

  while (i < len) {
    const ch = input[i]

    // ── Escape (top level) ──────────────────────
    if (escaped) {
      word += ch
      escaped = false
      i++
      continue
    }

    // ── Backtick mode ───────────────────────────
    if (backtick) {
      word += ch
      if (ch === '`') backtick = false
      i++
      continue
    }

    // ── $() mode ────────────────────────────────
    if (subDepth > 0) {
      word += ch

      if (subEscaped) {
        subEscaped = false
      } else if (subSingleQ) {
        if (ch === "'") subSingleQ = false
      } else if (subDoubleQ) {
        if (ch === '\\') {
          subEscaped = true
        } else if (ch === '"') {
          subDoubleQ = false
        } else if (ch === '$' && i + 1 < len && input[i + 1] === '(') {
          subDepth++
          word += input[i + 1]
          i += 2
          continue
        }
      } else {
        if (ch === '\\') subEscaped = true
        else if (ch === "'") subSingleQ = true
        else if (ch === '"') subDoubleQ = true
        else if (ch === '`') {
          // Backtick inside $() — just track for balancing
          // We accumulate as-is (already done above)
        }
        else if (ch === '(') subDepth++
        else if (ch === ')') subDepth--
      }

      i++
      continue
    }

    // ── Single-quote mode ───────────────────────
    if (singleQ) {
      if (ch === "'") {
        singleQ = false
      } else {
        word += ch
      }
      i++
      continue
    }

    // ── Double-quote mode ───────────────────────
    if (doubleQ) {
      if (ch === '\\') {
        escaped = true
        i++
        continue
      }
      if (ch === '"') {
        doubleQ = false
        i++
        continue
      }
      if (ch === '`') {
        word += ch
        backtick = true
        i++
        continue
      }
      if (ch === '$' && i + 1 < len && input[i + 1] === '(') {
        word += '$('
        subDepth = 1
        subSingleQ = false
        subDoubleQ = false
        subEscaped = false
        i += 2
        continue
      }
      word += ch
      i++
      continue
    }

    // ── Normal mode ─────────────────────────────

    // Backslash
    if (ch === '\\') {
      escaped = true
      i++
      continue
    }

    // Quote starts
    if (ch === "'") {
      singleQ = true
      i++
      continue
    }
    if (ch === '"') {
      doubleQ = true
      i++
      continue
    }
    if (ch === '`') {
      word += ch
      backtick = true
      i++
      continue
    }

    // $() start
    if (ch === '$' && i + 1 < len && input[i + 1] === '(') {
      word += '$('
      subDepth = 1
      subSingleQ = false
      subDoubleQ = false
      subEscaped = false
      i += 2
      continue
    }

    // Whitespace
    if (ch === ' ' || ch === '\t') {
      flush()
      i++
      continue
    }

    // Newline
    if (ch === '\n') {
      flush()
      tokens.push({ type: 'Semi' })
      i++
      continue
    }

    // Semicolon
    if (ch === ';') {
      flush()
      tokens.push({ type: 'Semi' })
      i++
      continue
    }

    // Pipe or Or
    if (ch === '|') {
      flush()
      if (i + 1 < len && input[i + 1] === '|') {
        tokens.push({ type: 'Or' })
        i += 2
      } else {
        tokens.push({ type: 'Pipe' })
        i++
      }
      continue
    }

    // And, &>, &>>, or background &
    if (ch === '&') {
      flush()
      if (i + 1 < len && input[i + 1] === '&') {
        tokens.push({ type: 'And' })
        i += 2
      } else if (i + 1 < len && input[i + 1] === '>') {
        if (i + 2 < len && input[i + 2] === '>') {
          tokens.push({ type: 'Redirect', op: '&>>' })
          i += 3
        } else {
          tokens.push({ type: 'Redirect', op: '&>' })
          i += 2
        }
      } else {
        // Background & — treat as separator
        tokens.push({ type: 'Semi' })
        i++
      }
      continue
    }

    // Redirect: >, >>, 2>, 2>>
    if (ch === '>') {
      const isFd2 = word === '2'
      if (isFd2) {
        word = ''
      } else {
        flush()
      }

      if (i + 1 < len && input[i + 1] === '>') {
        // >> or 2>> (both are append writes, map 2>> to >>)
        tokens.push({ type: 'Redirect', op: '>>' })
        i += 2
      } else {
        tokens.push({ type: 'Redirect', op: isFd2 ? '2>' : '>' })
        i++
      }
      continue
    }

    // Subshell open paren
    if (ch === '(') {
      flush()
      tokens.push({ type: 'Semi' })
      parenDepth++
      i++
      continue
    }

    // Subshell close paren
    if (ch === ')' && parenDepth > 0) {
      flush()
      tokens.push({ type: 'Semi' })
      parenDepth--
      i++
      continue
    }

    // Default: accumulate into word
    word += ch
    i++
  }

  flush()
  return tokens
}

// ─── Parser ──────────────────────────────────────────────────

export function parse(tokens: Token[]): SimpleCommand[] {
  const commands: SimpleCommand[] = []
  let start = 0

  for (let i = 0; i <= tokens.length; i++) {
    if (i === tokens.length || isSeparator(tokens[i])) {
      if (i > start) {
        const segment = tokens.slice(start, i)
        commands.push(parseSegment(segment))
      }
      start = i + 1
    }
  }

  return commands
}

function isSeparator(token: Token): boolean {
  const t = token.type
  return t === 'Pipe' || t === 'And' || t === 'Or' || t === 'Semi'
}

function parseSegment(tokens: Token[]): SimpleCommand {
  const assignments: Assignment[] = []
  const args: string[] = []
  const redirects: Redirect[] = []
  let name: string | null = null
  let foundCommand = false

  let i = 0
  while (i < tokens.length) {
    const tok = tokens[i]

    if (tok.type === 'Redirect') {
      const next = tokens[i + 1]
      if (next && next.type === 'Word') {
        redirects.push({ op: tok.op, target: next.value })
        i += 2
      } else {
        i++
      }
      continue
    }

    if (tok.type === 'Word') {
      if (!foundCommand && isAssignment(tok.value)) {
        const eq = tok.value.indexOf('=')
        assignments.push({
          name: tok.value.slice(0, eq),
          value: tok.value.slice(eq + 1)
        })
      } else if (!foundCommand) {
        name = tok.value
        foundCommand = true
      } else {
        args.push(tok.value)
      }
      i++
      continue
    }

    i++
  }

  return { assignments, name, args, redirects }
}

function isAssignment(word: string): boolean {
  const eq = word.indexOf('=')
  if (eq <= 0) return false
  for (let j = 0; j < eq; j++) {
    const c = word.charCodeAt(j)
    const isLetter = (c >= 65 && c <= 90) || (c >= 97 && c <= 122)
    const isUnderscore = c === 95
    if (j === 0) {
      if (!isLetter && !isUnderscore) return false
    } else {
      const isDigit = c >= 48 && c <= 57
      if (!isLetter && !isDigit && !isUnderscore) return false
    }
  }
  return true
}

// ─── Public API ──────────────────────────────────────────────

export function parseShellCommand(input: string): SimpleCommand[] {
  return parse(tokenize(input))
}
