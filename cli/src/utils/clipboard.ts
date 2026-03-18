import { closeSync, openSync, writeSync } from 'fs'
import { spawnSync } from 'child_process'
import { $ } from 'bun'
import { tmpdir } from 'os'
import path from 'path'
import { useRenderer } from '@opentui/react'
import { useEffect, useRef, useState } from 'react'
import { INPUT_CURSOR_CHAR } from '../components/multiline-input'
import { useMountedRef } from '../hooks/use-mounted-ref'
import { useSafeAsync } from '../hooks/use-safe-async'
import { useSafeTimeout } from '../hooks/use-safe-timeout'
import { safeRenderableAccess, safeRenderableCall } from './safe-renderable-access'

// =============================================================================
// Read from Clipboard (cross-platform)
// =============================================================================

/**
 * Read text from clipboard. Returns null if reading fails.
 */
export function readClipboardText(): string | null {
  try {
    const platform = process.platform
    const opts = { encoding: 'utf-8' as const, timeout: 1000 }

    let result
    switch (platform) {
      case 'darwin':
        result = spawnSync('pbpaste', [], opts)
        break
      case 'win32':
        result = spawnSync('powershell', ['-Command', 'Get-Clipboard'], opts)
        break
      case 'linux':
        result = spawnSync('xclip', ['-selection', 'clipboard', '-o'], opts)
        break
      default:
        return null
    }

    if (result.status === 0 && result.stdout) {
      return result.stdout.replace(/\n+$/, '')
    }
    return null
  } catch {
    return null
  }
}

// 32KB is safe for all environments (tmux is the strictest)
const OSC52_MAX_BASE64_PAYLOAD = 32_000

function inRemoteShell(): boolean {
  return !!(process.env.SSH_CLIENT || process.env.SSH_TTY || process.env.SSH_CONNECTION)
}

function createOsc52Payload(text: string): string | null {
  if (process.env.TERM === 'dumb') return null

  const base64 = Buffer.from(text, 'utf8').toString('base64')
  if (base64.length > OSC52_MAX_BASE64_PAYLOAD) return null

  const rawOsc52 = `\x1b]52;c;${base64}\x07`

  if (!process.env.TMUX && !process.env.STY) {
    return rawOsc52
  }

  if (process.env.TMUX) {
    return `\x1bPtmux;${rawOsc52.replace(/\x1b/g, '\x1b\x1b')}\x1b\\`
  }

  return `\x1bP${rawOsc52}\x1b\\`
}

function sendOsc52Copy(text: string): boolean {
  const sequence = createOsc52Payload(text)
  if (!sequence) return false

  const ttyPath = process.platform === 'win32' ? 'CON' : '/dev/tty'
  let fd: number | null = null
  try {
    fd = openSync(ttyPath, 'w')
    writeSync(fd, sequence)
    return true
  } catch {
    return false
  } finally {
    if (fd !== null) closeSync(fd)
  }
}

function copyUsingSystemClipboard(text: string): boolean {
  const opts = { input: text, stdio: ['pipe', 'ignore', 'ignore'] as ('pipe' | 'ignore')[] }

  try {
    if (process.platform === 'linux') {
      try {
        spawnSync('xclip', ['-selection', 'clipboard'], opts)
      } catch {
        spawnSync('xsel', ['--clipboard', '--input'], opts)
      }
    } else if (process.platform === 'win32') {
      spawnSync('clip', [], opts)
    } else if (process.platform === 'darwin') {
      spawnSync('pbcopy', [], opts)
    } else {
      return false
    }
    return true
  } catch {
    return false
  }
}

// =============================================================================
// Copy to Clipboard (cross-platform)
// =============================================================================

/**
 * Copy text to clipboard using platform-native tools or OSC52 escape sequence.
 * In remote sessions (SSH), prefers OSC52 which writes to the client terminal's clipboard.
 * In local sessions, prefers platform tools (pbcopy/xclip/clip).
 */
export async function writeTextToClipboard(text: string) {
  if (!text || text.trim().length === 0) {
    return
  }

  try {
    let copied: boolean
    if (inRemoteShell()) {
      // Remote/SSH: prefer OSC 52 (copies to client terminal's clipboard)
      copied = sendOsc52Copy(text) || copyUsingSystemClipboard(text)
    } else {
      // Local: prefer platform tools (reliable with tmux), OSC 52 as fallback
      copied = copyUsingSystemClipboard(text) || sendOsc52Copy(text)
    }

    if (!copied) {
      throw new Error('No clipboard method available')
    }
  } catch (error) {
    console.error('Failed to copy to clipboard', error)
    throw error
  }
}

export interface ClipboardBitmapResult {
  base64: string
  mime: string
  width: number
  height: number
}

/** Parse PNG/JPEG headers to extract dimensions without external dependencies */
export function extractImageDimensions(buffer: Buffer): { width: number; height: number } | null {
  // PNG: signature 89 50 4E 47, width at bytes 16-19, height at 20-23 (big-endian)
  if (buffer.length >= 24 && buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47) {
    const width = buffer.readUInt32BE(16)
    const height = buffer.readUInt32BE(20)
    return { width, height }
  }
  // JPEG: signature FF D8, scan for SOF0 (FFC0) or SOF2 (FFC2) marker
  if (buffer.length >= 4 && buffer[0] === 0xFF && buffer[1] === 0xD8) {
    let offset = 2
    while (offset < buffer.length - 9) {
      if (buffer[offset] !== 0xFF) break
      const marker = buffer[offset + 1]
      if (marker === 0xC0 || marker === 0xC2) {
        const height = buffer.readUInt16BE(offset + 5)
        const width = buffer.readUInt16BE(offset + 7)
        return { width, height }
      }
      const segLength = buffer.readUInt16BE(offset + 2)
      offset += 2 + segLength
    }
  }
  return null
}

export async function readClipboardBitmap(): Promise<ClipboardBitmapResult | null> {
  const platform = process.platform

  let base64: string | null = null
  let mime = 'image/png'

  if (platform === 'linux') {
    try {
      const wayland = await $`wl-paste -t image/png`.nothrow().arrayBuffer()
      if (wayland && wayland.byteLength > 0) {
        base64 = Buffer.from(wayland).toString('base64')
      }
    } catch {}
    if (!base64) {
      try {
        const x11 = await $`xclip -selection clipboard -t image/png -o`.nothrow().arrayBuffer()
        if (x11 && x11.byteLength > 0) {
          base64 = Buffer.from(x11).toString('base64')
        }
      } catch {}
    }
  } else if (platform === 'win32') {
    const script = "Add-Type -AssemblyName System.Windows.Forms; $img = [System.Windows.Forms.Clipboard]::GetImage(); if ($img) { $ms = New-Object System.IO.MemoryStream; $img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); [System.Convert]::ToBase64String($ms.ToArray()) }"
    try {
      const result = await $`powershell.exe -NonInteractive -NoProfile -command "${script}"`.nothrow().text()
      if (result) {
        const trimmed = result.trim()
        if (trimmed) base64 = trimmed
      }
    } catch {}
  } else if (platform === 'darwin') {
    const tmpfile = path.join(tmpdir(), 'magnitude-clipboard.png')
    try {
      await $`osascript -e 'set imageData to the clipboard as "PNGf"' -e 'set fileRef to open for access POSIX file "${tmpfile}" with write permission' -e 'set eof fileRef to 0' -e 'write imageData to fileRef' -e 'close access fileRef'`
        .nothrow()
        .quiet()
      const file = Bun.file(tmpfile)
      const buffer = await file.arrayBuffer()
      if (buffer.byteLength > 0) {
        base64 = Buffer.from(buffer).toString('base64')
      }
    } catch {}
    finally {
      await $`rm -f "${tmpfile}"`.nothrow().quiet()
    }
  }

  if (!base64) return null

  const dims = extractImageDimensions(Buffer.from(base64, 'base64'))
  if (!dims) return null

  return { base64, mime, width: dims.width, height: dims.height }
}

const COPY_DEBOUNCE_MS = 250
const TOAST_DURATION_MS = 3500

export function useSelectionAutoCopy() {
  const renderer = useRenderer()
  const mountedRef = useMountedRef()
  const safeTimeout = useSafeTimeout()
  const safeAsync = useSafeAsync()
  const [showCopiedToast, setShowCopiedToast] = useState(false)
  const copyDebounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const toastHideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingSelectionRef = useRef<string | null>(null)
  const lastCopiedSelectionRef = useRef<string | null>(null)

  useEffect(() => {
    const onSelectionChanged = (selectionEvent: any) => {
      const selectionObj = selectionEvent ?? safeRenderableAccess(
        renderer,
        (r) => r.getSelection?.(),
        { mountedRef, fallback: null },
      )
      const rawText: string | null = selectionObj?.getSelectedText
        ? selectionObj.getSelectedText()
        : typeof selectionObj === 'string'
          ? selectionObj
          : null

      const cleanedText = rawText?.replace(new RegExp(INPUT_CURSOR_CHAR, 'g'), '') ?? null

      if (!cleanedText || cleanedText.trim().length === 0) {
        pendingSelectionRef.current = null
        if (copyDebounceTimerRef.current) {
          safeTimeout.clear(copyDebounceTimerRef.current)
          copyDebounceTimerRef.current = null
        }
        return
      }

      if (cleanedText === pendingSelectionRef.current) return

      pendingSelectionRef.current = cleanedText

      if (copyDebounceTimerRef.current) safeTimeout.clear(copyDebounceTimerRef.current)

      copyDebounceTimerRef.current = safeTimeout.set(() => {
        copyDebounceTimerRef.current = null
        const pending = pendingSelectionRef.current
        if (!pending || pending === lastCopiedSelectionRef.current) return

        lastCopiedSelectionRef.current = pending
        void safeAsync.run(async (ctx) => {
          try {
            await writeTextToClipboard(pending)
            if (!ctx.checkpoint()) return
            safeRenderableCall(renderer, (r) => r.clearSelection(), {
              mountedRef,
            })
            setShowCopiedToast(true)
            if (toastHideTimerRef.current) safeTimeout.clear(toastHideTimerRef.current)
            toastHideTimerRef.current = safeTimeout.set(() => setShowCopiedToast(false), TOAST_DURATION_MS)
          } catch {
            // Errors logged within writeTextToClipboard
          }
        })
      }, COPY_DEBOUNCE_MS)
    }

    if (renderer?.on) {
      renderer.on('selection', onSelectionChanged)
      return () => {
        renderer.off?.('selection', onSelectionChanged)
      }
    }
    return undefined
  }, [mountedRef, renderer, safeAsync, safeTimeout])

  return { showCopiedToast }
}