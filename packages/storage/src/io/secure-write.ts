import { writeTextFile } from './fs'

export interface SecureWriteJsonOptions {
  readonly mode?: number
  readonly spaces?: number
  readonly appendNewline?: boolean
}

export async function writeSecureJsonFile(
  path: string,
  value: unknown,
  options?: SecureWriteJsonOptions
): Promise<void> {
  const content = JSON.stringify(value, null, options?.spaces ?? 2)

  await writeTextFile(path, content, {
    mode: options?.mode ?? 0o600,
    appendNewline: options?.appendNewline ?? true,
  })
}