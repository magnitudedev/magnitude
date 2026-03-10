import { mkdir, rm } from 'node:fs/promises'
import { join, basename } from 'node:path'
import { getTarget, getVersion } from './platform'

const REPO = 'microsoft/ripgrep-prebuilt'

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function fetchWithRetry(url: string, init: RequestInit, attempts = 3): Promise<Response> {
  let lastError: unknown
  for (let i = 0; i < attempts; i++) {
    try {
      const response = await fetch(url, init)
      if (!response.ok) throw new Error(`[ripgrep] HTTP ${response.status} for ${url}`)
      return response
    } catch (error) {
      lastError = error
      if (i < attempts - 1) await sleep(2 ** i * 1000)
    }
  }
  throw new Error(
    `[ripgrep] Download failed after ${attempts} attempts: ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }`
  )
}

/**
 * Download ripgrep binary into `destDir`.
 * Optionally specify a ripgrep target triple (e.g. 'aarch64-apple-darwin').
 * Defaults to current platform.
 * Returns the absolute path to the rg binary.
 */
export async function downloadRg(destDir: string, targetOverride?: string): Promise<string> {
  await mkdir(destDir, { recursive: true })

  const target = targetOverride ?? getTarget()
  const version = getVersion(target)
  const isWin = target.includes('windows')
  const ext = isWin ? '.zip' : '.tar.gz'
  const assetName = `ripgrep-${version}-${target}${ext}`
  const binName = isWin ? 'rg.exe' : 'rg'
  const binPath = join(destDir, binName)
  const token = process.env.GITHUB_TOKEN

  // Fetch release metadata
  const apiHeaders: Record<string, string> = { Accept: 'application/json' }
  if (token) apiHeaders.Authorization = `token ${token}`

  const releaseUrl = `https://api.github.com/repos/${REPO}/releases/tags/${version}`
  const releaseRes = await fetchWithRetry(releaseUrl, { headers: apiHeaders })
  const release = await releaseRes.json() as { assets?: Array<{ name: string; url: string }> }

  if (!release.assets) {
    throw new Error('[ripgrep] GitHub release missing assets. Check rate limits or token permissions.')
  }

  const asset = release.assets.find(a => a.name === assetName)
  if (!asset) throw new Error(`[ripgrep] Asset not found: ${assetName}`)

  // Download asset
  const dlHeaders: Record<string, string> = { Accept: 'application/octet-stream' }
  if (token) dlHeaders.Authorization = `token ${token}`

  const dlRes = await fetchWithRetry(asset.url, { headers: dlHeaders })
  const bytes = new Uint8Array(await dlRes.arrayBuffer())
  const tmpFile = join(destDir, `${assetName}.tmp`)

  try {
    await Bun.write(tmpFile, bytes)

    // Extract
    const tarArgs = isWin ? ['-xf', tmpFile, '-C', destDir] : ['-xzf', tmpFile, '-C', destDir]
    const proc = Bun.spawn(['tar', ...tarArgs], { stdout: 'ignore', stderr: 'pipe' })
    const exitCode = await proc.exited
    if (exitCode !== 0) {
      const stderr = await new Response(proc.stderr).text()
      throw new Error(`[ripgrep] tar failed (${exitCode}): ${stderr}`)
    }

    if (!await Bun.file(binPath).exists()) {
      throw new Error(`[ripgrep] ${basename(binPath)} not found after extraction`)
    }

    if (!isWin) {
      const ch = Bun.spawn(['chmod', '755', binPath], { stdout: 'ignore', stderr: 'ignore' })
      await ch.exited
    }

    return binPath
  } finally {
    await rm(tmpFile, { force: true }).catch(() => {})
  }
}