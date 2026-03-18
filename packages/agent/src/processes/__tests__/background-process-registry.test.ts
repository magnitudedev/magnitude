import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { Effect } from 'effect'
import { spawn, type ChildProcess } from 'child_process'
import { make } from '../background-process-registry'
import type { AppEvent } from '../../events'

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

async function waitFor<T>(fn: () => T | undefined, timeoutMs = 4000, intervalMs = 25): Promise<T> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const value = fn()
    if (value !== undefined) return value
    await wait(intervalMs)
  }
  throw new Error('Timed out waiting for condition')
}

async function waitUntil(fn: () => boolean, timeoutMs = 4000, intervalMs = 25): Promise<void> {
  await waitFor(() => (fn() ? true : undefined), timeoutMs, intervalMs)
}

describe('background process registry', () => {
  let events: AppEvent[] = []
  let registry: ReturnType<typeof make>
  let children: ChildProcess[] = []

  const publish = (event: AppEvent) => {
    events.push(event)
  }

  const trackSpawn = (command: string, args: string[]) => {
    const child = spawn(command, args)
    children.push(child)
    return child
  }

  beforeEach(() => {
    events = []
    registry = make(publish)
    children = []
  })

  afterEach(async () => {
    await Effect.runPromise(registry.shutdownAll())
    for (const child of children) {
      try {
        child.kill('SIGKILL')
      } catch {
        // ignore
      }
    }
  })

  test('register() publishes background_process_registered with correct fields', async () => {
    const child = trackSpawn('sleep', ['10'])
    const startedAt = Date.now()

    await Effect.runPromise(registry.register({
      pid: child.pid!,
      forkId: 'fork-a',
      turnId: 'turn-1',
      command: 'sleep 10',
      startedAt,
      child,
      initialStdout: 'hello',
      initialStderr: 'warn',
    }))

    const registered = events.find(
      (event): event is Extract<AppEvent, { type: 'background_process_registered' }> =>
        event.type === 'background_process_registered'
    )

    expect(registered).toBeDefined()
    expect(registered).toMatchObject({
      forkId: 'fork-a',
      pid: child.pid,
      command: 'sleep 10',
      sourceTurnId: 'turn-1',
      startedAt,
      initialStdout: 'hello',
      initialStderr: 'warn',
    })
  })

  test('register() with a process that quickly exits publishes registered and exited events', async () => {
    const child = trackSpawn('sleep', ['0.1'])

    await Effect.runPromise(registry.register({
      pid: child.pid!,
      forkId: 'fork-a',
      turnId: 'turn-1',
      command: 'sleep 0.1',
      startedAt: Date.now(),
      child,
      initialStdout: '',
      initialStderr: '',
    }))

    await waitUntil(() => events.some(event => event.type === 'background_process_exited'))

    expect(events.some(event => event.type === 'background_process_registered')).toBe(true)
    const exited = events.find(
      (event): event is Extract<AppEvent, { type: 'background_process_exited' }> =>
        event.type === 'background_process_exited'
    )
    expect(exited).toBeDefined()
    expect(exited?.pid).toBe(child.pid)
  })

  test('flush() publishes inline background_process_output events', async () => {
    const child = trackSpawn('bash', ['-lc', 'printf one; sleep 0.2; printf two; sleep 0.7'])

    await Effect.runPromise(registry.register({
      pid: child.pid!,
      forkId: 'fork-a',
      turnId: 'turn-1',
      command: 'printf one; sleep 0.2; printf two; sleep 0.7',
      startedAt: Date.now(),
      child,
      initialStdout: '',
      initialStderr: '',
    }))

    await waitUntil(() => {
      void Effect.runPromise(registry.flush('fork-a'))
      return events.some(
        event => event.type === 'background_process_output'
          && event.mode === 'inline'
          && event.stdoutChunk.includes('one')
      )
    })

    const output = events.find(
      (event): event is Extract<AppEvent, { type: 'background_process_output' }> =>
        event.type === 'background_process_output'
        && event.mode === 'inline'
        && event.stdoutChunk.includes('one')
    )!

    expect(output.mode).toBe('inline')
    if (output.mode !== 'inline') throw new Error('expected inline output')
    expect(output.stdoutChunk).toContain('one')
  })

  test('flush() demotes large buffered output and publishes demoted + tail events', async () => {
    const large = 'x'.repeat(9000)
    const child = trackSpawn('bash', ['-lc', `printf '${large}' ; sleep 1`])

    await Effect.runPromise(registry.register({
      pid: child.pid!,
      forkId: 'fork-a',
      turnId: 'turn-1',
      command: 'large output',
      startedAt: Date.now(),
      child,
      initialStdout: '',
      initialStderr: '',
    }))

    await waitUntil(() => {
      void Effect.runPromise(registry.flush('fork-a'))
      return events.some(event => event.type === 'background_process_demoted')
    })

    expect(events.some(event => event.type === 'background_process_demoted')).toBe(true)
    const output = events.find(
      (event): event is Extract<AppEvent, { type: 'background_process_output' }> =>
        event.type === 'background_process_output' && event.mode === 'tail'
    )
    expect(output).toBeDefined()
    if (!output || output.mode !== 'tail') throw new Error('expected tail output')
    expect(output.stdoutLines).toBe(0)
  })

  test('listByFork() returns only processes for the given fork', async () => {
    const childA = trackSpawn('sleep', ['10'])
    const childB = trackSpawn('sleep', ['10'])

    await Effect.runPromise(registry.register({
      pid: childA.pid!,
      forkId: 'fork-a',
      turnId: 'turn-1',
      command: 'sleep 10',
      startedAt: Date.now(),
      child: childA,
      initialStdout: '',
      initialStderr: '',
    }))
    await Effect.runPromise(registry.register({
      pid: childB.pid!,
      forkId: 'fork-b',
      turnId: 'turn-2',
      command: 'sleep 10',
      startedAt: Date.now(),
      child: childB,
      initialStdout: '',
      initialStderr: '',
    }))

    const records = await Effect.runPromise(registry.listByFork('fork-a'))
    expect(records.map(record => record.pid)).toEqual([childA.pid!])
  })

  test('getByPid() returns the correct record', async () => {
    const child = trackSpawn('sleep', ['10'])

    await Effect.runPromise(registry.register({
      pid: child.pid!,
      forkId: 'fork-a',
      turnId: 'turn-1',
      command: 'sleep 10',
      startedAt: Date.now(),
      child,
      initialStdout: '',
      initialStderr: '',
    }))

    const record = await Effect.runPromise(registry.getByPid(child.pid!))
    expect(record).toBeDefined()
    expect(record?.forkId).toBe('fork-a')
    expect(record?.command).toBe('sleep 10')
  })

  test('cleanupFork() kills processes owned by that fork', async () => {
    const owned = trackSpawn('sleep', ['10'])
    const other = trackSpawn('sleep', ['10'])

    await Effect.runPromise(registry.register({
      pid: owned.pid!,
      forkId: 'fork-a',
      turnId: 'turn-1',
      command: 'sleep 10',
      startedAt: Date.now(),
      child: owned,
      initialStdout: '',
      initialStderr: '',
    }))
    await Effect.runPromise(registry.register({
      pid: other.pid!,
      forkId: 'fork-b',
      turnId: 'turn-2',
      command: 'sleep 10',
      startedAt: Date.now(),
      child: other,
      initialStdout: '',
      initialStderr: '',
    }))

    await Effect.runPromise(registry.cleanupFork('fork-a'))
    await waitUntil(() =>
      events.some(
        event => event.type === 'background_process_exited' && event.pid === owned.pid
      )
    )

    const ownedRecord = await Effect.runPromise(registry.getByPid(owned.pid!))
    const otherRecord = await Effect.runPromise(registry.getByPid(other.pid!))

    expect(ownedRecord?.status).toBe('exited')
    expect(otherRecord?.status).toBe('running')
  })

  test('shutdownAll() kills all tracked processes', async () => {
    const childA = trackSpawn('sleep', ['10'])
    const childB = trackSpawn('sleep', ['10'])

    for (const [child, forkId] of [[childA, 'fork-a'], [childB, 'fork-b']] as const) {
      await Effect.runPromise(registry.register({
        pid: child.pid!,
        forkId,
        turnId: `turn-${forkId}`,
        command: 'sleep 10',
        startedAt: Date.now(),
        child,
        initialStdout: '',
        initialStderr: '',
      }))
    }

    await Effect.runPromise(registry.shutdownAll())
    expect(await Effect.runPromise(registry.getByPid(childA.pid!))).toBeUndefined()
    expect(await Effect.runPromise(registry.getByPid(childB.pid!))).toBeUndefined()
  })

  test('exit detection publishes background_process_exited with correct exitCode/signal', async () => {
    const child = trackSpawn('bash', ['-lc', 'exit 7'])

    await Effect.runPromise(registry.register({
      pid: child.pid!,
      forkId: 'fork-a',
      turnId: 'turn-1',
      command: 'exit 7',
      startedAt: Date.now(),
      child,
      initialStdout: '',
      initialStderr: '',
    }))

    const exited = await waitFor(() =>
      events.find(
        (event): event is Extract<AppEvent, { type: 'background_process_exited' }> =>
          event.type === 'background_process_exited' && event.pid === child.pid
      )
    )

    expect(exited.exitCode).toBe(7)
    expect(exited.signal).toBeNull()
    expect(exited.status).toBe('exited')
  })
})