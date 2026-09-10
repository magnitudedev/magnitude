import { chmod, mkdir, rename } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { isWindows } from './platform'

// Extracted executables are keyed by the embedded payload's digest, not the
// upstream version: a re-signed rg with the same version must never reuse a
// previously extracted unsigned copy. Each release therefore owns an immutable
// file, and an old release still running keeps its own.
const RIPGREP_DIR = join(homedir(), '.magnitude', 'bin', 'ripgrep')

let cachedPath: string | null = null
let resolvePromise: Promise<string> | null = null

async function embeddedPayload(): Promise<Uint8Array> {
  // Dynamic import so this is only resolved at runtime, not during
  // workspace builds or bundling that would try to parse the binary as JS.
  const { rgPath } = await import('./rg-embed')
  const file = Bun.file(rgPath)
  if (!await file.exists()) {
    throw new Error(
      '[ripgrep] Packaging invariant violated: ripgrep binary not found. ' +
      'This binary was built incorrectly.'
    )
  }
  return new Uint8Array(await file.arrayBuffer())
}

async function extractEmbedded(): Promise<string> {
  const bytes = await embeddedPayload()
  const digest = createHash('sha256').update(bytes).digest('hex')
  const directory = join(RIPGREP_DIR, digest)
  const binPath = join(directory, isWindows() ? 'rg.exe' : 'rg')
  if (await Bun.file(binPath).exists()) return binPath

  await mkdir(directory, { recursive: true })
  // Publish by rename: an executable is never rewritten in place while a
  // concurrent process may be running it. Competing publishers write identical bytes.
  const temporary = join(directory, `.rg-${process.pid}-${Date.now()}`)
  await Bun.write(temporary, bytes)
  if (!isWindows()) await chmod(temporary, 0o755)
  try {
    await rename(temporary, binPath)
  } catch (error) {
    if (!await Bun.file(binPath).exists()) throw error
  }
  return binPath
}

/**
 * Resolve the path to the ripgrep binary.
 * Uses the extracted binary for this exact embedded payload if present,
 * otherwise extracts it. No download fallback — missing rg is a build failure.
 */
export async function resolveRgPath(): Promise<string> {
  if (cachedPath) return cachedPath

  if (!resolvePromise) {
    resolvePromise = extractEmbedded().then(path => {
      cachedPath = path
      return path
    }).finally(() => {
      resolvePromise = null
    })
  }

  return resolvePromise
}
