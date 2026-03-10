import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { getTarget, getVersion, isWindows } from './platform'
import { downloadRg } from './download'

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

function getEmbeddedRg(): Blob | undefined {
  const files = (Bun as any).embeddedFiles as Array<Blob & { name?: string }> | undefined
  if (!files?.length) return undefined
  return files.find(f => f.name?.endsWith('/rg') || f.name?.endsWith('/rg.exe'))
}

async function extractEmbedded(): Promise<string> {
  const blob = getEmbeddedRg()
  if (!blob) throw new Error('[ripgrep] No embedded binary available')

  await mkdir(BIN_DIR, { recursive: true })
  const binPath = getRgBinPath()
  await Bun.write(binPath, blob)

  if (!isWindows()) {
    const proc = Bun.spawn(['chmod', '755', binPath], { stdout: 'ignore', stderr: 'ignore' })
    await proc.exited
  }

  await Bun.write(VERSION_MARKER, versionString())
  return binPath
}

async function install(): Promise<string> {
  // Try embedded first (compiled binary mode)
  if (getEmbeddedRg()) {
    return await extractEmbedded()
  }

  // Dev mode: download
  const binPath = await downloadRg(BIN_DIR)
  await Bun.write(VERSION_MARKER, versionString())
  return binPath
}

/**
 * Resolve the path to the ripgrep binary.
 * Uses cached binary if available, otherwise extracts from embedded (compiled) or downloads (dev).
 */
export async function resolveRgPath(): Promise<string> {
  if (cachedPath) return cachedPath

  const binPath = getRgBinPath()
  if (await Bun.file(binPath).exists() && await versionMatches()) {
    cachedPath = binPath
    return binPath
  }

  if (!resolvePromise) {
    resolvePromise = install().then(path => {
      cachedPath = path
      return path
    }).finally(() => {
      resolvePromise = null
    })
  }

  return resolvePromise
}