import { appendFile, mkdir, readdir, rm } from 'node:fs/promises'
import { dirname, join } from 'node:path'

export interface WriteTextOptions {
  readonly mode?: number
  readonly appendNewline?: boolean
}

export async function pathExists(path: string): Promise<boolean> {
  return Bun.file(path).exists()
}

export async function ensureDir(dirPath: string): Promise<void> {
  await mkdir(dirPath, { recursive: true })
}

export async function ensureParentDir(filePath: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true })
}

export async function readTextFile(path: string): Promise<string> {
  return Bun.file(path).text()
}

export async function writeTextFile(
  path: string,
  content: string,
  options?: WriteTextOptions
): Promise<void> {
  await ensureParentDir(path)

  let nextContent = content
  if (options?.appendNewline && !nextContent.endsWith('\n')) {
    nextContent += '\n'
  }

  await Bun.write(path, nextContent, options?.mode != null ? { mode: options.mode } : undefined)
}

export async function appendTextFile(
  path: string,
  content: string
): Promise<void> {
  await ensureParentDir(path)
  await appendFile(path, content)
}

export async function removeFileIfExists(path: string): Promise<void> {
  try {
    await rm(path, { force: false })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return
    }

    throw error
  }
}

export async function listDirectory(
  path: string
): Promise<Array<{
  name: string
  path: string
  isFile: boolean
  isDirectory: boolean
}>> {
  const entries = await readdir(path, { withFileTypes: true })

  return entries.map((entry) => ({
    name: entry.name,
    path: join(path, entry.name),
    isFile: entry.isFile(),
    isDirectory: entry.isDirectory(),
  }))
}