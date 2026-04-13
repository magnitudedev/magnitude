import { closeSync, openSync, writeSync } from 'fs'
import * as childProcess from 'child_process'
import type { SpawnSyncReturns } from 'child_process'
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
// Clipboard strategy
// =============================================================================
//
// Clipboard behavior is centralized in this module for both copy and read paths.
// Input/paste ingestion is already centralized elsewhere (SingleLineInput /
// MultilineInput + paste coordinator), so reliability improvements are localized
// here and automatically propagate across those paths.
//

type ClipboardPlatform = NodeJS.Platform
type ClipboardWriteBackend = 'osc52' | 'pbcopy' | 'clip' | 'powershell' | 'wl-copy' | 'xclip' | 'xsel'
type ClipboardReadTextBackend = 'pbpaste' | 'powershell-get' | 'wl-paste' | 'xclip-read' | 'xsel-read'
type ClipboardReadImageBackend = 'wl-paste-image' | 'xclip-image' | 'powershell-image' | 'osascript-image'

export interface ClipboardEnv {
  platform: ClipboardPlatform
  remoteShell: boolean
  isWsl: boolean
  hasWayland: boolean
  hasX11: boolean
  inTmux: boolean
  inScreen: boolean
  term: string | undefined
}

interface CommandResult {
  success: boolean
  stdout: string
}

interface CommandRunnerOptions {
  input?: string
  timeout?: number
  encoding?: BufferEncoding
}

function detectClipboardEnv(
  platform: ClipboardPlatform = process.platform,
  env: NodeJS.ProcessEnv = process.env,
): ClipboardEnv {
  return {
    platform,
    remoteShell: Boolean(env.SSH_CLIENT || env.SSH_TTY || env.SSH_CONNECTION),
    isWsl: platform === 'linux' && Boolean(env.WSL_DISTRO_NAME || env.WSL_INTEROP),
    hasWayland: Boolean(env.WAYLAND_DISPLAY),
    hasX11: Boolean(env.DISPLAY),
    inTmux: Boolean(env.TMUX),
    inScreen: Boolean(env.STY),
    term: env.TERM,
  }
}

function commandSucceeded(result: Pick<SpawnSyncReturns<string>, 'status' | 'error'>): boolean {
  return result.status === 0 && !result.error
}

function runClipboardCommand(
  command: string,
  args: string[],
  options: CommandRunnerOptions = {},
): CommandResult {
  const result = childProcess.spawnSync(command, args, {
    input: options.input,
    timeout: options.timeout ?? 1000,
    encoding: options.encoding ?? 'utf-8',
    stdio: options.input === undefined ? ['ignore', 'pipe', 'ignore'] : ['pipe', 'pipe', 'ignore'],
  }) as SpawnSyncReturns<string>

  const success = commandSucceeded(result)
  return { success, stdout: result.stdout ?? '' }
}

function selectWriteBackends(env: ClipboardEnv): ClipboardWriteBackend[] {
  const backends: ClipboardWriteBackend[] = ['osc52']

  switch (env.platform) {
    case 'darwin':
      backends.push('pbcopy')
      break
    case 'win32':
      backends.push('clip', 'powershell')
      break
    case 'linux':
      if (env.hasWayland) backends.push('wl-copy')
      backends.push('xclip', 'xsel')
      break
  }

  return backends
}

function selectReadTextBackends(env: ClipboardEnv): ClipboardReadTextBackend[] {
  switch (env.platform) {
    case 'darwin':
      return ['pbpaste']
    case 'win32':
      return ['powershell-get']
    case 'linux':
      if (env.isWsl) return ['powershell-get', 'wl-paste', 'xclip-read', 'xsel-read']
      if (env.hasWayland) return ['wl-paste', 'xclip-read', 'xsel-read']
      return ['xclip-read', 'xsel-read', 'wl-paste']
    default:
      return []
  }
}

function selectReadImageBackends(env: ClipboardEnv): ClipboardReadImageBackend[] {
  switch (env.platform) {
    case 'darwin':
      return ['osascript-image']
    case 'win32':
      return ['powershell-image']
    case 'linux':
      if (env.isWsl) return ['powershell-image', 'wl-paste-image', 'xclip-image']
      return env.hasWayland ? ['wl-paste-image', 'xclip-image'] : ['xclip-image', 'wl-paste-image']
    default:
      return []
  }
}

// 32KB is safe for all environments (tmux is the strictest)
const OSC52_MAX_BASE64_PAYLOAD = 32_000

function createOsc52Payload(text: string, env: ClipboardEnv): string | null {
  if (env.term === 'dumb') return null

  const base64 = Buffer.from(text, 'utf8').toString('base64')
  if (base64.length > OSC52_MAX_BASE64_PAYLOAD) return null

  const rawOsc52 = `\x1b]52;c;${base64}\x07`

  if (!env.inTmux && !env.inScreen) return rawOsc52
  if (env.inTmux) return `\x1bPtmux;${rawOsc52.replace(/\x1b/g, '\x1b\x1b')}\x1b\\`
  return `\x1bP${rawOsc52}\x1b\\`
}

function tryOsc52Write(text: string, env: ClipboardEnv): boolean {
  const sequence = createOsc52Payload(text, env)
  if (!sequence) return false

  const ttyPath = env.platform === 'win32' ? 'CON' : '/dev/tty'
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

function tryWriteBackend(backend: ClipboardWriteBackend, text: string, env: ClipboardEnv): boolean {
  switch (backend) {
    case 'osc52':
      return tryOsc52Write(text, env)
    case 'pbcopy':
      return runClipboardCommand('pbcopy', [], { input: text }).success
    case 'clip':
      return runClipboardCommand('clip', [], { input: text }).success
    case 'powershell':
      return runClipboardCommand('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', 'Set-Clipboard'], { input: text }).success
    case 'wl-copy':
      return runClipboardCommand('wl-copy', [], { input: text }).success
    case 'xclip':
      return runClipboardCommand('xclip', ['-selection', 'clipboard'], { input: text }).success
    case 'xsel':
      return runClipboardCommand('xsel', ['--clipboard', '--input'], { input: text }).success
  }
}

function tryReadTextBackend(backend: ClipboardReadTextBackend): string | null {
  let result: CommandResult | null = null
  switch (backend) {
    case 'pbpaste':
      result = runClipboardCommand('pbpaste', [])
      break
    case 'powershell-get':
      result = runClipboardCommand('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard'])
      break
    case 'wl-paste':
      result = runClipboardCommand('wl-paste', ['-n'])
      break
    case 'xclip-read':
      result = runClipboardCommand('xclip', ['-selection', 'clipboard', '-o'])
      break
    case 'xsel-read':
      result = runClipboardCommand('xsel', ['--clipboard', '--output'])
      break
  }

  if (!result?.success) return null
  const value = result.stdout.replace(/\n+$/, '')
  return value.length > 0 ? value : null
}

/**
 * Read text from clipboard. Returns null if reading fails.
 */
export function readClipboardText(): string | null {
  try {
    const env = detectClipboardEnv()
    for (const backend of selectReadTextBackends(env)) {
      const value = tryReadTextBackend(backend)
      if (value) return value
    }
    return null
  } catch {
    return null
  }
}

/**
 * Copy text to clipboard using platform backends and OSC52 fallback.
 * Throws when all eligible backends fail.
 */
export async function writeTextToClipboard(text: string) {
  if (!text || text.trim().length === 0) return

  const env = detectClipboardEnv()
  for (const backend of selectWriteBackends(env)) {
    if (tryWriteBackend(backend, text, env)) return
  }

  const error = new Error('No clipboard backend succeeded')
  console.error('Failed to copy to clipboard', error)
  throw error
}

export interface ClipboardBitmapResult {
  base64: string
  mime: string
  width: number
  height: number
}

/** Parse PNG/JPEG headers to extract dimensions without external dependencies */
export function extractImageDimensions(buffer: Buffer): { width: number; height: number } | null {
  if (buffer.length >= 24 && buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47) {
    const width = buffer.readUInt32BE(16)
    const height = buffer.readUInt32BE(20)
    return { width, height }
  }
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

async function tryReadImageBackend(backend: ClipboardReadImageBackend): Promise<string | null> {
  switch (backend) {
    case 'wl-paste-image': {
      try {
        const wayland = await $`wl-paste -t image/png`.nothrow().arrayBuffer()
        if (wayland && wayland.byteLength > 0) return Buffer.from(wayland).toString('base64')
      } catch {}
      return null
    }
    case 'xclip-image': {
      try {
        const x11 = await $`xclip -selection clipboard -t image/png -o`.nothrow().arrayBuffer()
        if (x11 && x11.byteLength > 0) return Buffer.from(x11).toString('base64')
      } catch {}
      return null
    }
    case 'powershell-image': {
      const script = "Add-Type -AssemblyName System.Windows.Forms; $img = [System.Windows.Forms.Clipboard]::GetImage(); if ($img) { $ms = New-Object System.IO.MemoryStream; $img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); [System.Convert]::ToBase64String($ms.ToArray()) }"
      try {
        const result = await $`powershell.exe -NonInteractive -NoProfile -command "${script}"`.nothrow().text()
        const trimmed = result?.trim()
        return trimmed ? trimmed : null
      } catch {
        return null
      }
    }
    case 'osascript-image': {
      const tmpfile = path.join(tmpdir(), 'magnitude-clipboard.png')
      try {
        await $`osascript -e 'set imageData to the clipboard as "PNGf"' -e 'set fileRef to open for access POSIX file "${tmpfile}" with write permission' -e 'set eof fileRef to 0' -e 'write imageData to fileRef' -e 'close access fileRef'`
          .nothrow()
          .quiet()
        const buffer = await Bun.file(tmpfile).arrayBuffer()
        if (buffer.byteLength > 0) return Buffer.from(buffer).toString('base64')
      } catch {
      } finally {
        await $`rm -f "${tmpfile}"`.nothrow().quiet()
      }
      return null
    }
  }
}

export async function readClipboardBitmap(): Promise<ClipboardBitmapResult | null> {
  const env = detectClipboardEnv()
  const mime = 'image/png'

  for (const backend of selectReadImageBackends(env)) {
    const base64 = await tryReadImageBackend(backend)
    if (!base64) continue
    const dims = extractImageDimensions(Buffer.from(base64, 'base64'))
    if (!dims) continue
    return { base64, mime, width: dims.width, height: dims.height }
  }

  return null
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

export const __clipboardInternals = {
  detectClipboardEnv,
  selectWriteBackends,
  selectReadTextBackends,
  selectReadImageBackends,
  createOsc52Payload,
  runClipboardCommand,
  commandSucceeded,
}