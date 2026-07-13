export function hasFlag(args: readonly string[], ...flags: string[]): boolean {
  const flagSet = new Set(flags)
  return args.some(arg => flagSet.has(arg))
}

export function hasAnyToken(args: readonly string[], tokens: ReadonlySet<string>): boolean {
  return args.some(arg => tokens.has(arg))
}

export function normalizeArgs(args: readonly string[]): readonly string[] {
  return args.map(arg => arg.toLowerCase())
}

export function cloudCommandPath(args: readonly string[], flagsWithValue: ReadonlySet<string>): readonly string[] {
  const path: string[] = []

  for (let i = 0; i < args.length; i++) {
    const token = args[i]
    if (token === '--') continue

    if (token.startsWith('--')) {
      const eqIndex = token.indexOf('=')
      if (eqIndex >= 0) {
        const flag = token.slice(0, eqIndex)
        if (flagsWithValue.has(flag)) continue
      } else if (flagsWithValue.has(token)) {
        i += 1
        continue
      }
    } else if (token.startsWith('-') && token.length > 2) {
      const shortFlag = token.slice(0, 2)
      if (flagsWithValue.has(shortFlag)) continue
    } else if (flagsWithValue.has(token)) {
      i += 1
      continue
    }

    path.push(token)
  }

  return path
}

export function hasSubcommand(path: readonly string[], ...segments: string[]): boolean {
  if (segments.length > path.length) return false
  for (let i = 0; i < segments.length; i++) {
    if (path[i] !== segments[i]) return false
  }
  return true
}