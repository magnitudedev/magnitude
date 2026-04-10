import { lstat, readlink, rm, symlink } from 'node:fs/promises'
import { basename, extname, join, resolve } from 'node:path'

import { generateSortableId } from '@magnitudedev/generate-id'

import {
  appendJsonLines,
  ensureDir,
  listDirectory,
  readJsonFile,
  readJsonFileWithSchema,
  readJsonLines,
  removeFileIfExists,
  writeJsonFile,
} from '../io'
import type { GlobalStoragePaths } from '../paths/global-paths'
import {
  MemoryExtractionJobRecordSchema,
  type MemoryExtractionJobRecord,
  type SessionDiscoveryOptions,
  type StoredSessionMeta,
} from '../types/session'

const TIMESTAMP_SESSION_ID_RE = /^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$/
const SORTABLE_SESSION_ID_RE = /^[0-9a-z]{6,12}$/

export function createTimestampSessionId(): string {
  return generateSortableId()
}

export async function listSessionIds(
  paths: GlobalStoragePaths,
  options?: SessionDiscoveryOptions
): Promise<string[]> {
  const timestampOnly = options?.timestampOnly ?? true

  try {
    const entries = await listDirectory(paths.sessionsRoot)

    return entries
      .filter((entry) => entry.isDirectory)
      .map((entry) => entry.name)
      .filter((name) => !timestampOnly || TIMESTAMP_SESSION_ID_RE.test(name) || SORTABLE_SESSION_ID_RE.test(name))
      .sort()
      .reverse()
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return []
    }
    throw error
  }
}

export async function findLatestSessionId(
  paths: GlobalStoragePaths,
  options?: SessionDiscoveryOptions
): Promise<string | null> {
  const sessionIds = await listSessionIds(paths, options)
  return sessionIds[0] ?? null
}

export async function readRawSessionMeta(
  paths: GlobalStoragePaths,
  sessionId: string
): Promise<unknown | null> {
  return readJsonFile<unknown | null>(paths.sessionMetaFile(sessionId), { fallback: null })
}

export async function writeSessionMeta(
  paths: GlobalStoragePaths,
  sessionId: string,
  meta: StoredSessionMeta
): Promise<void> {
  await ensureDir(paths.sessionDir(sessionId))
  await writeJsonFile(paths.sessionMetaFile(sessionId), meta)
}

export async function updateSessionMeta(
  paths: GlobalStoragePaths,
  sessionId: string,
  current: StoredSessionMeta | null,
  updater: (current: StoredSessionMeta | null) => StoredSessionMeta
): Promise<StoredSessionMeta> {
  const next = updater(current)
  await writeSessionMeta(paths, sessionId, next)
  return next
}

export async function readSessionEvents<T>(
  paths: GlobalStoragePaths,
  sessionId: string
): Promise<T[]> {
  return readJsonLines<T>(paths.sessionEventsFile(sessionId))
}

export async function appendSessionEvents<T>(
  paths: GlobalStoragePaths,
  sessionId: string,
  events: readonly T[]
): Promise<void> {
  await ensureDir(paths.sessionDir(sessionId))
  await appendJsonLines(paths.sessionEventsFile(sessionId), events)
}

export async function readSessionEventsFromPath<T>(
  eventsPath: string
): Promise<T[]> {
  return readJsonLines<T>(eventsPath)
}

export async function createSessionWorkspace(
  paths: GlobalStoragePaths,
  sessionId: string,
  cwd: string
): Promise<string> {
  const workspacePath = paths.sessionWorkspace(sessionId)
  const projectLinkPath = join(workspacePath, 'project')

  await ensureDir(workspacePath)

  try {
    const stat = await lstat(projectLinkPath)
    if (stat.isSymbolicLink()) {
      const existingTarget = await readlink(projectLinkPath)
      const resolvedTarget = resolve(workspacePath, existingTarget)
      if (resolvedTarget === cwd) {
        return workspacePath
      }
    }
    await rm(projectLinkPath, { recursive: true, force: true })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }
  }

  await symlink(cwd, projectLinkPath, process.platform === 'win32' ? 'junction' : 'dir')
  return workspacePath
}

export function createMemoryExtractionJobRecord(params: {
  sessionId: string
  cwd: string
  eventsPath: string
  memoryPath: string
  now?: Date
  createId?: () => string
}): MemoryExtractionJobRecord {
  const now = params.now ?? new Date()
  const uniqueId = params.createId?.() ?? `${Math.random().toString(36).slice(2)}`
  return {
    jobId: `${params.sessionId}-${now.getTime()}-${uniqueId}`,
    sessionId: params.sessionId,
    cwd: params.cwd,
    eventsPath: params.eventsPath,
    memoryPath: params.memoryPath,
    createdAt: now.toISOString(),
    attempts: 0,
    status: 'pending',
  }
}

export function resolvePendingMemoryJobPath(
  paths: GlobalStoragePaths,
  jobId: string
): string {
  return paths.pendingMemoryJobFile(jobId)
}

export async function writePendingMemoryJob(
  paths: GlobalStoragePaths,
  job: MemoryExtractionJobRecord
): Promise<string> {
  const filePath = resolvePendingMemoryJobPath(paths, job.jobId)
  await ensureDir(paths.pendingMemoryExtractionRoot)
  await writeJsonFile(filePath, job)
  return filePath
}

export async function listPendingMemoryJobFiles(
  paths: GlobalStoragePaths
): Promise<string[]> {
  try {
    const entries = await listDirectory(paths.pendingMemoryExtractionRoot)
    return entries
      .filter((entry) => entry.isFile && extname(entry.name) === '.json')
      .map((entry) => entry.path)
      .sort()
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return []
    }
    throw error
  }
}

export async function listPendingMemoryJobIds(
  paths: GlobalStoragePaths
): Promise<string[]> {
  const files = await listPendingMemoryJobFiles(paths)
  return files
    .map((filePath) => basename(filePath, '.json'))
    .sort()
}

function getPendingMemoryJobPath(
  paths: GlobalStoragePaths,
  input: { readonly jobId: string } | { readonly filePath: string }
): string {
  return 'jobId' in input
    ? resolvePendingMemoryJobPath(paths, input.jobId)
    : input.filePath
}

export async function readPendingMemoryJob(
  paths: GlobalStoragePaths,
  input: { readonly jobId: string } | { readonly filePath: string }
): Promise<MemoryExtractionJobRecord> {
  return readJsonFileWithSchema(
    getPendingMemoryJobPath(paths, input),
    MemoryExtractionJobRecordSchema
  )
}

export async function markPendingMemoryJobRunning(
  paths: GlobalStoragePaths,
  input: { readonly jobId: string } | { readonly filePath: string },
  job?: MemoryExtractionJobRecord
): Promise<MemoryExtractionJobRecord> {
  const current = job ?? await readPendingMemoryJob(paths, input)
  const next: MemoryExtractionJobRecord = {
    ...current,
    status: 'running',
    attempts: (current.attempts ?? 0) + 1,
  }
  await writeJsonFile(getPendingMemoryJobPath(paths, input), next)
  return next
}

export async function markPendingMemoryJobPending(
  paths: GlobalStoragePaths,
  input: { readonly jobId: string } | { readonly filePath: string },
  job?: MemoryExtractionJobRecord
): Promise<MemoryExtractionJobRecord> {
  const current = job ?? await readPendingMemoryJob(paths, input)
  const next: MemoryExtractionJobRecord = {
    ...current,
    status: 'pending',
  }
  await writeJsonFile(getPendingMemoryJobPath(paths, input), next)
  return next
}

export async function removePendingMemoryJob(
  paths: GlobalStoragePaths,
  input: { readonly jobId: string } | { readonly filePath: string }
): Promise<void> {
  await removeFileIfExists(getPendingMemoryJobPath(paths, input))
}