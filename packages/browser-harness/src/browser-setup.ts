/**
 * Browser Setup Detection & Installation
 *
 * Detects whether patchright's Chromium binary is installed and provides
 * installation support. Uses patchright's own registry to determine the
 * expected binary path.
 */

import { chromium } from 'playwright'  // aliased to patchright
import { accessSync } from 'fs'

/**
 * Check if the Chromium binary required by patchright is installed.
 * Uses chromium.executablePath() to get the expected path, then verifies
 * the file exists and is accessible.
 */
export function isBrowserInstalled(): boolean {
  try {
    const execPath = chromium.executablePath()
    if (!execPath) return false
    accessSync(execPath)
    return true
  } catch {
    return false
  }
}

/**
 * Get the expected Chromium executable path (even if not installed).
 * Returns null if the path cannot be determined.
 */
export function getBrowserExecutablePath(): string | null {
  try {
    return chromium.executablePath() || null
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
