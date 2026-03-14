import { appendTextFile, readTextFile } from './fs'

export async function readJsonLines<T>(path: string): Promise<T[]> {
  try {
    const raw = await readTextFile(path)

    return raw
      .split('\n')
      .filter((line) => line.trim() !== '')
      .map((line) => JSON.parse(line) as T)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return []
    }

    throw error
  }
}

export async function appendJsonLines<T>(
  path: string,
  entries: ReadonlyArray<T>
): Promise<void> {
  if (entries.length === 0) {
    return
  }

  const content = entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n'
  await appendTextFile(path, content)
}