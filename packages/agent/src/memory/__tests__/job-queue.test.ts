import { describe, test, expect, beforeEach, afterEach } from 'bun:test'
import { mkdtempSync } from 'fs'
import { rm } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import {
  createMemoryExtractionJob,
  writePendingJobSync,
  listPendingJobs,
  readJob,
  removeJob,
  getPendingDir,
} from '../job-queue'

describe('memory job queue', () => {
  const origHome = process.env.HOME
  let homeDir = ''

  beforeEach(() => {
    homeDir = mkdtempSync(join(tmpdir(), 'mem-queue-home-'))
    process.env.HOME = homeDir
  })

  afterEach(async () => {
    await rm(homeDir, { recursive: true, force: true })
    process.env.HOME = origHome
  })

  test('sync marker write + list/read/remove', async () => {
    const job = createMemoryExtractionJob({
      sessionId: 's1',
      cwd: '/tmp/project',
      eventsPath: '/tmp/events.jsonl',
      memoryPath: '/tmp/project/.magnitude/memory.md',
    })

    const jobPath = writePendingJobSync(job)
    expect(jobPath.startsWith(getPendingDir())).toBe(true)

    const files = await listPendingJobs()
    expect(files.length).toBe(1)

    const roundTrip = await readJob(files[0]!)
    expect(roundTrip.sessionId).toBe('s1')
    expect(roundTrip.status).toBe('pending')

    await removeJob(files[0]!)
    expect((await listPendingJobs()).length).toBe(0)
  })
})