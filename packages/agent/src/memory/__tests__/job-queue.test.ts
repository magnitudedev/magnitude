import { describe, test, expect, beforeEach, afterEach } from 'bun:test'
import { mkdtempSync } from 'fs'
import { rm } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { createStorageClient, type StorageClient } from '@magnitudedev/storage'
import { createMemoryExtractionJob, writePendingJob } from '../job-queue'

describe('memory job queue', () => {
  const origHome = process.env.HOME
  let homeDir = ''
  let storage: StorageClient

  beforeEach(async () => {
    homeDir = mkdtempSync(join(tmpdir(), 'mem-queue-home-'))
    process.env.HOME = homeDir
    storage = await createStorageClient()
  })

  afterEach(async () => {
    await rm(homeDir, { recursive: true, force: true })
    process.env.HOME = origHome
  })

  test('write + list/read/remove', async () => {
    const job = createMemoryExtractionJob({
      sessionId: 's1',
      cwd: '/tmp/project',
      eventsPath: '/tmp/events.jsonl',
      memoryPath: '/tmp/project/.magnitude/memory.md',
    })

    const jobPath = await writePendingJob(storage, job)
    expect(jobPath.length).toBeGreaterThan(0)

    const files = await storage.memoryJobs.list()
    expect(files.length).toBe(1)

    const roundTrip = await storage.memoryJobs.read({ filePath: files[0]! })
    expect(roundTrip.sessionId).toBe('s1')
    expect(roundTrip.status).toBe('pending')

    await storage.memoryJobs.remove(roundTrip.jobId)
    expect((await storage.memoryJobs.list()).length).toBe(0)
  })
})