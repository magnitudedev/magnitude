import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { getTarget, getVersion, isWindows } from './platform'

const BIN_DIR = join(homedir(), '.magnitude', 'bin')
const VERSION_MARKER = join(BIN_DIR, 'rg.version')

let cachedPath: string | null = null
let resolvePromise: Promise<string> | null = null

function getRgBinPath(): string {
  return join(BIN_DIR, isWindows() ? 'rg.exe' : 'rg')
}

function versionString(): string {
  const target = getTarget()
  return `${getVersion(target)}|${target}`
}

async function versionMatches(): Promise<boolean> {
  try {
    return (await Bun.file(VERSION_MARKER).text()).trim() === versionString()
  } catch {
    return false
  }
}

async function getRgPath(): Promise<string> {
  // Dynamic import so this is only resolved at runtime, not during
  // workspace builds or bundling that would try to parse the binary as JS.
  const { rgPath } = await import('./rg-embed')
  return rgPath
}

async function extractEmbedded(): Promise<string> {
  const rgPath = await getRgPath()
  const file = Bun.file(rgPath)
  if (!await file.exists()) {
    throw new Error(
      '[ripgrep] Packaging invariant violated: ripgrep binary not found. ' +
      'This binary was built incorrectly.'
    )
  }

  await mkdir(BIN_DIR, { recursive: true })
  const binPath = getRgBinPath()
  await Bun.write(binPath, file)

  if (!isWindows()) {
    const proc = Bun.spawn(['chmod', '755', binPath], { stdout: 'ignore', stderr: 'ignore' })
    await proc.exited
  }

  await Bun.write(VERSION_MARKER, versionString())
  return binPath
}

/**
 * Resolve the path to the ripgrep binary.
 * Uses cached binary if available, otherwise extracts from the embedded binary.
 * No download fallback — missing rg is a packaging/build failure.
 */
export async function resolveRgPath(): Promise<string> {
  if (cachedPath) return cachedPath

  const binPath = getRgBinPath()
  if (await Bun.file(binPath).exists() && await versionMatches()) {
    cachedPath = binPath
    return binPath
  }

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
