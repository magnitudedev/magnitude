import {
  appendFileSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { dirname } from 'node:path'

export function appendJsonLinesSync<T>(
  filePath: string,
  entries: readonly T[]
): void {
  if (entries.length === 0) {
    return
  }

  mkdirSync(dirname(filePath), { recursive: true })
  appendFileSync(
    filePath,
    entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n',
    'utf-8'
  )
}

export function clearFileSync(filePath: string): void {
  try {
    unlinkSync(filePath)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }
  }
}

export function readJsonFileSync<T>(filePath: string, fallback: T): T {
  try {
    return JSON.parse(readFileSync(filePath, 'utf-8')) as T
  } catch {
    return fallback
  }
}

export function writeJsonFileSync(
  filePath: string,
  value: unknown,
  options?: { readonly mode?: number }
): void {
  mkdirSync(dirname(filePath), { recursive: true })

  let content = JSON.stringify(value, null, 2)
  if (!content.endsWith('\n')) {
    content += '\n'
  }

  writeFileSync(filePath, content, {
    encoding: 'utf-8',
    ...(options?.mode !== undefined ? { mode: options.mode } : {}),
  })
}

export function writeSecureJsonFileSync(filePath: string, value: unknown): void {
  writeJsonFileSync(filePath, value, { mode: 0o600 })
}