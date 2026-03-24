/**
 * Browser Setup Detection & Installation
 *
 * Detects whether patchright's Chromium binary is installed and provides
 * installation support. Uses patchright's own registry to determine the
 * expected binary path.
 */

import { chromium } from 'playwright'  // aliased to patchright
import { accessSync, readdirSync } from 'fs'
import { homedir } from 'os'
import { join } from 'path'

function getMsPlaywrightCacheDir(): string {
  if (process.platform === 'win32') {
    return join(process.env.LOCALAPPDATA ?? join(homedir(), 'AppData', 'Local'), 'ms-playwright')
  } else if (process.platform === 'darwin') {
    return join(homedir(), 'Library', 'Caches', 'ms-playwright')
  } else {
    return join(process.env.XDG_CACHE_HOME ?? join(homedir(), '.cache'), 'ms-playwright')
  }
}

/**
 * Check if any patchright-cached Chromium binary is available.
 * Accepts any chromium-* revision, not just the exact bundled one.
 */
export function isBrowserInstalled(): boolean {
  try {
    const cacheDir = getMsPlaywrightCacheDir()
    const entries = readdirSync(cacheDir)
    return entries.some(e => e.startsWith('chromium-'))
  } catch {
    return false
  }
}

/**
 * Get the Chromium executable path, preferring the exact bundled revision
 * but falling back to any other cached chromium-* revision.
 * Returns null if no usable Chromium binary is found.
 */
export function getBrowserExecutablePath(): string | null {
  try {
    // First try the exact bundled path
    const bundledPath = chromium.executablePath()
    if (bundledPath) {
      try { accessSync(bundledPath); return bundledPath } catch {}
    }

    // Collect non-headless chromium dirs sorted by revision (highest first)
    const cacheDir = getMsPlaywrightCacheDir()
    const entries = readdirSync(cacheDir)
    const chromiumDirs = entries
      .filter(e => e.startsWith('chromium-') && !e.includes('headless'))
      .sort((a, b) => {
        const revA = parseInt(a.split('-')[1]) || 0
        const revB = parseInt(b.split('-')[1]) || 0
        return revB - revA  // highest first
      })

    if (chromiumDirs.length === 0) return null

    // If bundled path has a recognizable revision dir, use it as a template
    if (bundledPath) {
      const bundledRevDir = bundledPath.match(/chromium-\d+/)?.[0]
      if (bundledRevDir) {
        for (const dir of chromiumDirs) {
          const candidate = bundledPath.replace(bundledRevDir, dir)
          try { accessSync(candidate); return candidate } catch {}
        }
      }
    }

    // Fallback: walk each chromium dir directly using known platform-specific paths
    for (const dir of chromiumDirs) {
      try {
        const chromiumPath = join(cacheDir, dir)
        const subEntries = readdirSync(chromiumPath)
        const chromeSubDir = subEntries.find(e => e.startsWith('chrome-'))
        if (!chromeSubDir) continue

        const subPath = join(chromiumPath, chromeSubDir)
        let candidate: string
        if (process.platform === 'darwin') {
          candidate = join(subPath, 'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing')
        } else if (process.platform === 'win32') {
          candidate = join(subPath, 'chrome.exe')
        } else {
          candidate = join(subPath, 'chrome')
        }
        try { accessSync(candidate); return candidate } catch {}
      } catch {}
    }

    return null
  } catch {
    return null
  }
}

/**
 * Install the Chromium browser binary using patchright's installer.
 * Spawns `npx patchright install chromium` as a child process.
 *
 * @param onData - Optional callback for streaming stdout/stderr output
 * @returns Promise with success status and combined output
 */
export async function installBrowser(
  onData?: (chunk: string) => void
): Promise<{ success: boolean; output: string }> {
  const chunks: string[] = []

  return new Promise((resolve) => {
    const proc = Bun.spawn(['npx', 'patchright', 'install', 'chromium'], {
      stdout: 'pipe',
      stderr: 'pipe',
      env: { ...process.env },
    })

    const readStream = async (stream: ReadableStream<Uint8Array>) => {
      const reader = stream.getReader()
      const decoder = new TextDecoder()
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const text = decoder.decode(value, { stream: true })
          chunks.push(text)
          onData?.(text)
        }
      } catch {
        // Stream ended
      }
    }

    Promise.all([
      readStream(proc.stdout),
      readStream(proc.stderr),
    ]).then(async () => {
      const exitCode = await proc.exited
      resolve({
        success: exitCode === 0,
        output: chunks.join(''),
      })
    })
  })
}

/**
 * Get a human-readable description of what the browser install will do.
 */
export function getInstallDescription(): {
  command: string
  explanation: string
  estimatedSize: string
} {
  return {
    command: 'npx patchright install chromium',
    explanation: 'Downloads a Chromium browser binary required for the browser agent. This is a one-time setup.',
    estimatedSize: '~200MB',
  }
}
