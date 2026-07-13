import { describe, expect, test } from 'bun:test'
import { __clipboardInternals } from './clipboard'

describe('clipboard backend selection', () => {
  test('linux wayland tries osc52 first, then wl-copy/xclip/xsel', () => {
    const env = __clipboardInternals.detectClipboardEnv('linux', { WAYLAND_DISPLAY: 'wayland-0' } as NodeJS.ProcessEnv)
    expect(__clipboardInternals.selectWriteBackends(env)).toEqual(['osc52', 'wl-copy', 'xclip', 'xsel'])
  })

  test('remote linux keeps osc52 first', () => {
    const env = __clipboardInternals.detectClipboardEnv('linux', {
      WAYLAND_DISPLAY: 'wayland-0',
      SSH_CONNECTION: 'x',
    } as NodeJS.ProcessEnv)
    expect(__clipboardInternals.selectWriteBackends(env)).toEqual(['osc52', 'wl-copy', 'xclip', 'xsel'])
  })

  test('darwin tries osc52 before pbcopy', () => {
    const env = __clipboardInternals.detectClipboardEnv('darwin', {} as NodeJS.ProcessEnv)
    expect(__clipboardInternals.selectWriteBackends(env)).toEqual(['osc52', 'pbcopy'])
  })

  test('wsl text read includes powershell first', () => {
    const env = __clipboardInternals.detectClipboardEnv('linux', {
      WSL_DISTRO_NAME: 'Ubuntu',
    } as NodeJS.ProcessEnv)
    expect(__clipboardInternals.selectReadTextBackends(env)[0]).toBe('powershell-get')
  })
})

describe('clipboard write attempts', () => {
  test('continues to native backend after osc52 succeeds', () => {
    const env = __clipboardInternals.detectClipboardEnv('darwin', {} as NodeJS.ProcessEnv)
    const calls: string[] = []

    const copied = __clipboardInternals.attemptClipboardWrites('copy me', env, (backend) => {
      calls.push(backend)
      return backend === 'osc52' || backend === 'pbcopy'
    })

    expect(copied).toBe(true)
    expect(calls).toEqual(['osc52', 'pbcopy'])
  })

  test('falls through native backend chain until one succeeds', () => {
    const env = __clipboardInternals.detectClipboardEnv('linux', { WAYLAND_DISPLAY: 'wayland-0' } as NodeJS.ProcessEnv)
    const calls: string[] = []

    const copied = __clipboardInternals.attemptClipboardWrites('copy me', env, (backend) => {
      calls.push(backend)
      return backend === 'xclip'
    })

    expect(copied).toBe(true)
    expect(calls).toEqual(['osc52', 'wl-copy', 'xclip'])
  })

  test('reports false when no backend succeeds', () => {
    const env = __clipboardInternals.detectClipboardEnv('darwin', {} as NodeJS.ProcessEnv)
    const copied = __clipboardInternals.attemptClipboardWrites('copy me', env, () => false)

    expect(copied).toBe(false)
  })
})

describe('clipboard command success validation', () => {
  test('returns false on non-zero status', () => {
    expect(__clipboardInternals.commandSucceeded({ status: 1, error: undefined } as any)).toBe(false)
  })

  test('returns false when error is present', () => {
    expect(__clipboardInternals.commandSucceeded({ status: 0, error: new Error('boom') } as any)).toBe(false)
  })

  test('returns true only for status 0 without error', () => {
    expect(__clipboardInternals.commandSucceeded({ status: 0, error: undefined } as any)).toBe(true)
  })
})
