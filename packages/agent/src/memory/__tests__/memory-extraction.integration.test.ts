import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'
import { mkdtempSync } from 'fs'
import { mkdir, readFile, rm, writeFile } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'
import { createStorageClient, type StorageClient } from '@magnitudedev/storage'
import type { MagnitudeSlot } from '../../model-slots'
import { createMemoryExtractionJob, drainPendingJobsOnStartup, writePendingJob } from '../job-queue'
import { MEMORY_RELATIVE_PATH, ensureMemoryFile, readMemory } from '../memory-file'

type MockDiffResult = {
  result: unknown
  usage?: unknown
}

const providerState: {
  initCalls: number
  extractCalls: Array<{ transcript: string; currentMemory: string }>
  nextResult: MockDiffResult
  throwOnExtract: Error | null
} = {
  initCalls: 0,
  extractCalls: [],
  nextResult: { result: { additions: [], updates: [], deletions: [] } },
  throwOnExtract: null,
}

mock.module('@magnitudedev/providers', () => ({
  initializeProviderState: async () => {
    providerState.initCalls += 1
  },
  secondary: {
    extractMemoryDiff: async (transcript: string, currentMemory: string) => {
      providerState.extractCalls.push({ transcript, currentMemory })
      if (providerState.throwOnExtract) throw providerState.throwOnExtract
      return providerState.nextResult
    },
  },
}))

async function runExtractionJobFromFile(jobFilePath: string): Promise<void> {
  const mod = await import('../run-extraction-job')
  await mod.runExtractionJobFromFile(jobFilePath)
}

function toJsonl(events: unknown[]): string {
  return events.map((e) => JSON.stringify(e)).join('\n') + '\n'
}

function userEvent(text: string) {
  return {
    type: 'user_message',
    forkId: null,
    content: [{ type: 'text', text }],
    attachments: [],
    mode: 'text',
    synthetic: false,
    taskMode: false,
    timestamp: new Date().toISOString(),
  }
}

describe('memory extraction integration', () => {
  const originalHome = process.env.HOME
  let homeDir = ''
  let cwd = ''
  let storage: StorageClient<MagnitudeSlot>

  beforeEach(async () => {
    homeDir = mkdtempSync(join(tmpdir(), 'mem-extract-home-'))
    cwd = mkdtempSync(join(tmpdir(), 'mem-extract-cwd-'))
    process.env.HOME = homeDir

    providerState.initCalls = 0
    providerState.extractCalls = []
    providerState.throwOnExtract = null
    providerState.nextResult = { result: { additions: [], updates: [], deletions: [] } }

    storage = await createStorageClient<MagnitudeSlot>({ cwd })
    const jobs = await storage.memoryJobs.list()
    await Promise.all(jobs.map((filePath) => storage.memoryJobs.read({ filePath }).then((job) => storage.memoryJobs.remove(job.jobId))))
    await ensureMemoryFile(storage)
  })

  afterEach(async () => {
    const jobs = await storage.memoryJobs.list()
    await Promise.all(jobs.map((filePath) => storage.memoryJobs.read({ filePath }).then((job) => storage.memoryJobs.remove(job.jobId))))
    await rm(homeDir, { recursive: true, force: true })
    await rm(cwd, { recursive: true, force: true })
    process.env.HOME = originalHome
  })

  test('1) dispose wrapper writes marker correctly and synchronously', async () => {
    const sessionId = 'dispose-sync-session'
    let finishDispose: () => void = () => {}
    const originalDispose = new Promise<void>((resolve) => {
      finishDispose = resolve
    })

    const wrappedDisposeLikeCodingAgent = async () => {
      const eventsPath = join(process.env.HOME || '', '.magnitude', 'sessions', sessionId, 'events.jsonl')
      const memoryPath = join(cwd, MEMORY_RELATIVE_PATH)
      const job = createMemoryExtractionJob({ sessionId, cwd, eventsPath, memoryPath })
      const jobPath = await writePendingJob(storage, job)

      await originalDispose
      return jobPath
    }

    const disposing = wrappedDisposeLikeCodingAgent()
    const filesWhileDisposeBlocked = await storage.memoryJobs.list()

    expect(filesWhileDisposeBlocked.length).toBeGreaterThanOrEqual(1)
    const jobs = await Promise.all(filesWhileDisposeBlocked.map((filePath: string) => storage.memoryJobs.read({ filePath })))
    const job = jobs.find((j: { sessionId: string }) => j.sessionId === sessionId)
    expect(job).toBeTruthy()
    expect(job!.sessionId).toBe(sessionId)
    expect(job!.cwd).toBe(cwd)
    expect(job!.eventsPath).toBe(join(homeDir, '.magnitude', 'sessions', sessionId, 'events.jsonl'))
    expect(job!.memoryPath).toBe(join(cwd, MEMORY_RELATIVE_PATH))
    expect(job!.status).toBe('pending')

    finishDispose()
    await disposing
  })

  test('2) startup drain processes pending jobs', async () => {
    const sessionId = 'startup-drain-session'
    const eventsPath = join(homeDir, '.magnitude', 'sessions', sessionId, 'events.jsonl')
    await mkdir(join(homeDir, '.magnitude', 'sessions', sessionId), { recursive: true })
    await writeFile(eventsPath, toJsonl([userEvent('No, always use named exports.')]), 'utf8')

    providerState.nextResult = {
      result: {
        additions: [{ category: 'codebase', content: 'always use named exports' }],
        updates: [],
        deletions: [],
      },
    }

    const memoryPath = join(cwd, MEMORY_RELATIVE_PATH)
    const jobPath = await writePendingJob(storage, createMemoryExtractionJob({ sessionId, cwd, eventsPath, memoryPath }))

    await drainPendingJobsOnStartup(storage)

    const updated = await readMemory(storage)
    expect(updated).toContain('always use named exports')
    await expect(readFile(jobPath, 'utf8')).rejects.toThrow()
  })

  test('3) full extraction flow with mocked model', async () => {
    const sessionId = 'full-flow-session'
    const eventsPath = join(homeDir, '.magnitude', 'sessions', sessionId, 'events.jsonl')
    await mkdir(join(homeDir, '.magnitude', 'sessions', sessionId), { recursive: true })
    await writeFile(eventsPath, toJsonl([
      userEvent('no, always use named exports'),
      userEvent('always use an explorer before planning'),
      userEvent('please implement the endpoint'),
    ]), 'utf8')

    providerState.nextResult = {
      result: {
        additions: [
          { category: 'codebase', content: 'always use named exports' },
          { category: 'workflow', content: 'always use an explorer before planning' },
        ],
        updates: [],
        deletions: [],
      },
    }

    const jobPath = await writePendingJob(storage, createMemoryExtractionJob({
      sessionId,
      cwd,
      eventsPath,
      memoryPath: join(cwd, MEMORY_RELATIVE_PATH),
    }))

    await runExtractionJobFromFile(jobPath)

    const updated = await readMemory(storage)
    expect(updated).toContain('always use named exports')
    expect(updated).toContain('always use an explorer before planning')
    expect(updated).toContain('# Codebase')
    expect(updated).toContain('# Workflow')
  })

  test('4) empty diff leaves memory unchanged', async () => {
    const sessionId = 'empty-diff-session'
    const eventsPath = join(homeDir, '.magnitude', 'sessions', sessionId, 'events.jsonl')
    await mkdir(join(homeDir, '.magnitude', 'sessions', sessionId), { recursive: true })
    await writeFile(eventsPath, toJsonl([userEvent('implement this small refactor')]), 'utf8')

    providerState.nextResult = { result: { additions: [], updates: [], deletions: [] } }
    const before = await readMemory(storage)

    const jobPath = await writePendingJob(storage, createMemoryExtractionJob({
      sessionId,
      cwd,
      eventsPath,
      memoryPath: join(cwd, MEMORY_RELATIVE_PATH),
    }))

    await runExtractionJobFromFile(jobPath)

    const after = await readMemory(storage)
    expect(after).toBe(before)
  })

  test('5) invalid model JSON does not corrupt memory and does not throw', async () => {
    const sessionId = 'invalid-json-session'
    const eventsPath = join(homeDir, '.magnitude', 'sessions', sessionId, 'events.jsonl')
    await mkdir(join(homeDir, '.magnitude', 'sessions', sessionId), { recursive: true })
    await writeFile(eventsPath, toJsonl([userEvent('routine coding')]), 'utf8')

    providerState.nextResult = { result: 'not-json-object' as unknown }
    const before = await readMemory(storage)

    const jobPath = await writePendingJob(storage, createMemoryExtractionJob({
      sessionId,
      cwd,
      eventsPath,
      memoryPath: join(cwd, MEMORY_RELATIVE_PATH),
    }))

    await expect(runExtractionJobFromFile(jobPath)).resolves.toBeUndefined()
    const after = await readMemory(storage)
    expect(after).toBe(before)

    const pendingJob = await storage.memoryJobs.read({ filePath: jobPath })
    expect(pendingJob.status).toBe('pending')
  })

  test('6) model call failure keeps memory unchanged and leaves job for retry', async () => {
    const sessionId = 'model-failure-session'
    const eventsPath = join(homeDir, '.magnitude', 'sessions', sessionId, 'events.jsonl')
    await mkdir(join(homeDir, '.magnitude', 'sessions', sessionId), { recursive: true })
    await writeFile(eventsPath, toJsonl([userEvent('routine coding')]), 'utf8')

    providerState.throwOnExtract = new Error('model unavailable')
    const before = await readMemory(storage)

    const jobPath = await writePendingJob(storage, createMemoryExtractionJob({
      sessionId,
      cwd,
      eventsPath,
      memoryPath: join(cwd, MEMORY_RELATIVE_PATH),
    }))

    await expect(runExtractionJobFromFile(jobPath)).resolves.toBeUndefined()

    const after = await readMemory(storage)
    expect(after).toBe(before)

    const pendingJob = await storage.memoryJobs.read({ filePath: jobPath })
    expect(pendingJob.status).toBe('pending')
  })

  test('7) idempotence: running same diff twice does not duplicate entries', async () => {
    const sessionId = 'idempotence-session'
    const eventsPath = join(homeDir, '.magnitude', 'sessions', sessionId, 'events.jsonl')
    await mkdir(join(homeDir, '.magnitude', 'sessions', sessionId), { recursive: true })
    await writeFile(eventsPath, toJsonl([userEvent('no, always use named exports')]), 'utf8')

    providerState.nextResult = {
      result: {
        additions: [{ category: 'codebase', content: 'always use named exports' }],
        updates: [],
        deletions: [],
      },
    }

    const job1 = await writePendingJob(storage, createMemoryExtractionJob({
      sessionId,
      cwd,
      eventsPath,
      memoryPath: join(cwd, MEMORY_RELATIVE_PATH),
    }))
    await runExtractionJobFromFile(job1)
    const first = await readMemory(storage)

    const job2 = await writePendingJob(storage, createMemoryExtractionJob({
      sessionId,
      cwd,
      eventsPath,
      memoryPath: join(cwd, MEMORY_RELATIVE_PATH),
    }))
    await runExtractionJobFromFile(job2)
    const second = await readMemory(storage)

    expect(second).toBe(first)
    const count = second.split('\n').filter((l) => l.includes('always use named exports')).length
    expect(count).toBe(1)
  })


})