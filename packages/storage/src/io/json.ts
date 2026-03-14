import { Schema } from 'effect'

import { writeTextFile } from './fs'

export interface ReadJsonOptions<T> {
  readonly fallback?: T
}

export interface WriteJsonOptions {
  readonly mode?: number
  readonly appendNewline?: boolean
  readonly spaces?: number
}

export async function readJsonFile<T>(
  path: string,
  options?: ReadJsonOptions<T>
): Promise<T> {
  try {
    return await Bun.file(path).json() as T
  } catch (error) {
    if (
      options &&
      'fallback' in options &&
      ((error as NodeJS.ErrnoException).code === 'ENOENT' ||
        error instanceof SyntaxError)
    ) {
      return options.fallback as T
    }

    throw error
  }
}

export async function readJsonFileWithSchema<A, I>(
  path: string,
  schema: Schema.Schema<A, I>,
  options?: { readonly fallback?: A }
): Promise<A> {
  try {
    const raw = await Bun.file(path).json()
    return Schema.decodeUnknownSync(schema)(raw)
  } catch (error) {
    if (options && 'fallback' in options) {
      return options.fallback as A
    }
    throw error
  }
}

export async function writeJsonFile(
  path: string,
  value: unknown,
  options?: WriteJsonOptions
): Promise<void> {
  const content = JSON.stringify(value, null, options?.spaces ?? 2)

  await writeTextFile(path, content, {
    mode: options?.mode,
    appendNewline: options?.appendNewline ?? true,
  })
}

export async function updateJsonFile<T>(
  path: string,
  fallback: T,
  updater: (current: T) => T | Promise<T>,
  options?: WriteJsonOptions
): Promise<T> {
  const current = await readJsonFile<T>(path, { fallback })
  const next = await updater(current)
  await writeJsonFile(path, next, options)
  return next
}