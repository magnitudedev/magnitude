/**
 * ps-based process tree utilities.
 *
 * Provides a shared, platform-aware interface to `ps` for:
 * - Discovering all descendant PIDs of a process (BFS via ppid tree)
 * - Sampling CPU/memory metrics for a process and its descendants
 *
 * Platforms without `ps` (Windows) gracefully degrade:
 * - `getDescendantPids` returns an empty array (caller falls back to
 *   killing only the root PID)
 * - `sampleMetrics` returns an empty array (no metrics available)
 * A warning is logged once per session when `ps` is unavailable.
 *
 * Uses `Bun.spawn` (the native Bun subprocess API), not Node's
 * `child_process.spawn`. The reason: `child_process.spawn` reports a
 * missing-command failure as an unhandled `'error'` event on the
 * ChildProcess EventEmitter — which, under Bun, is dispatched before
 * any `process.nextTick` listener can attach, making the failure
 * uncatchable by `try/catch` or `Effect.catchAllCause` and crashing
 * the host process with exit 1. `Bun.spawn` throws a JS exception
 * synchronously on missing command, which Effect catches normally.
 * See bugs/26-06-21/ps-spawn-enoent-leak.md for the full diagnosis.
 */

import { Cause, Data, Effect } from 'effect'
import { logger } from '@magnitudedev/logger'

// ── Types ─────────────────────────────────────────────────────────────

export interface PsRow {
  readonly pid: number
  readonly ppid: number
  readonly cpu: number
  readonly rss: number // bytes
}

export interface ProcessMetricsSample {
  readonly pid: number
  readonly cpuPercent: number
  readonly rssBytes: number
  readonly timestamp: number
}

// ── Platform Detection ────────────────────────────────────────────────

/**
 * Returns the `ps` arguments for the current platform, or null if `ps`
 * is not available (Windows).
 *
 * macOS: `ps -axo pid,ppid,pcpu,rss` (header included, skipped during parse)
 * Linux: `ps -axo pid,ppid,%cpu,rss --no-headers`
 */
function getPsArgs(): string[] | null {
  if (process.platform === 'darwin') {
    return ['-axo', 'pid,ppid,pcpu,rss']
  }
  if (process.platform === 'linux') {
    return ['-axo', 'pid,ppid,%cpu,rss', '--no-headers']
  }
  return null
}

let psUnavailableWarned = false

function warnPsUnavailable() {
  if (psUnavailableWarned) return
  psUnavailableWarned = true
  logger.warn(
    { platform: process.platform },
    '[ps-tree] ps is not available on this platform — process tree discovery and metrics are disabled',
  )
}

// ── ps Invocation ─────────────────────────────────────────────────────

const PS_TIMEOUT = '2 seconds'

class PsInvocationError extends Data.TaggedError('PsInvocationError')<{
  readonly cause: unknown
}> {}

/**
 * Run `ps` and return its stdout as a string.
 * Returns null if `ps` is unavailable on this platform, exits non-zero,
 * times out, or fails to spawn (e.g. ENOENT — `ps` not in `$PATH`).
 *
 * All failure modes — including spawn-time ENOENT — are caught by
 * `Effect.catchAllCause` and degrade to `null`. This works because
 * `Bun.spawn` throws a JS exception on missing command (unlike Node's
 * `child_process.spawn`, which raises an unhandled EventEmitter
 * `'error'` event that escapes all JS-level catches — see the module
 * doc comment and bugs/26-06-21/ps-spawn-enoent-leak.md).
 */
function runPs(): Effect.Effect<string | null, never, never> {
  const args = getPsArgs()
  if (args === null) {
    warnPsUnavailable()
    return Effect.succeed(null)
  }

  return Effect.tryPromise({
    try: async () => {
      const proc = Bun.spawn({
        cmd: ['ps', ...args],
        stdin: 'ignore',
        stdout: 'pipe',
        stderr: 'pipe',
      })

      const stdoutText = await new Response(proc.stdout).text()
      const exitCode = await proc.exited

      if (exitCode !== 0) {
        const stderrText = await new Response(proc.stderr).text()
        logger.warn(
          { exitCode, stderr: stderrText.trim() },
          '[ps-tree] ps exited non-zero',
        )
        return null
      }

      return stdoutText
    },
    catch: (cause) => new PsInvocationError({ cause }),
  }).pipe(
    Effect.timeout(PS_TIMEOUT),
    Effect.catchAllCause((cause) =>
      Effect.gen(function* () {
        logger.warn(
          { cause: Cause.pretty(cause) },
          '[ps-tree] ps invocation failed',
        )
        return null
      }),
    ),
  )
}

// ── Parsing ───────────────────────────────────────────────────────────

/**
 * Parse `ps` output into a table of rows keyed by PID.
 * Handles platform differences (macOS includes a header line).
 */
export function parsePsOutput(
  output: string,
  platform: NodeJS.Platform = process.platform,
): Map<number, PsRow> {
  const table = new Map<number, PsRow>()
  let skipHeader = platform === 'darwin'

  for (const line of output.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed) continue

    if (skipHeader) {
      skipHeader = false
      continue
    }

    const parts = trimmed.split(/\s+/)
    if (parts.length < 4) continue

    const pid = parseInt(parts[0]!, 10)
    const ppid = parseInt(parts[1]!, 10)
    const cpu = parseFloat(parts[2]!)
    const rss = parseInt(parts[3]!, 10) * 1024 // KB → bytes

    if (Number.isNaN(pid) || Number.isNaN(ppid) || Number.isNaN(cpu) || Number.isNaN(rss)) {
      continue
    }

    table.set(pid, { pid, ppid, cpu, rss })
  }

  return table
}

/**
 * Build a ppid → children[] index from a PID table.
 */
export function buildChildrenIndex(table: Map<number, PsRow>): Map<number, number[]> {
  const children = new Map<number, number[]>()
  for (const row of table.values()) {
    const list = children.get(row.ppid)
    if (list) list.push(row.pid)
    else children.set(row.ppid, [row.pid])
  }
  return children
}

/**
 * BFS from a root PID, collecting all descendant PIDs (excluding root).
 */
export function getDescendantPids(
  rootPid: number,
  table: Map<number, PsRow>,
  children: Map<number, number[]>,
): number[] {
  const descendants: number[] = []
  const seen = new Set<number>([rootPid])
  const queue: number[] = [rootPid]

  while (queue.length > 0) {
    const pid = queue.shift()!
    const kids = children.get(pid)
    if (!kids) continue

    for (const kid of kids) {
      if (seen.has(kid)) continue
      seen.add(kid)
      descendants.push(kid)
      queue.push(kid)
    }
  }

  return descendants
}

// ── Public API ────────────────────────────────────────────────────────

/**
 * Discover all descendant PIDs of a root process.
 *
 * Returns an empty array if `ps` is unavailable (Windows) or the root
 * PID is not found in the process table. Callers should fall back to
 * killing only the root PID in that case.
 */
export function discoverDescendants(
  rootPid: number,
): Effect.Effect<number[], never, never> {
  return Effect.gen(function* () {
    const output = yield* runPs()
    if (output === null) return []

    const table = parsePsOutput(output)
    if (!table.has(rootPid)) return []

    const children = buildChildrenIndex(table)
    return getDescendantPids(rootPid, table, children)
  })
}

/**
 * Sample CPU and memory metrics for a set of root PIDs, including all
 * descendants.
 *
 * Returns an empty array if `ps` is unavailable (Windows) or no tracked
 * PIDs are found in the process table.
 */
export function sampleMetrics(
  rootPids: readonly number[],
): Effect.Effect<ProcessMetricsSample[], never, never> {
  return Effect.gen(function* () {
    if (rootPids.length === 0) return []

    const output = yield* runPs()
    if (output === null) return []

    const table = parsePsOutput(output)
    const children = buildChildrenIndex(table)
    const timestamp = Date.now()

    const samples: ProcessMetricsSample[] = []
    for (const rootPid of rootPids) {
      if (!table.has(rootPid)) continue

      let cpu = 0
      let rss = 0
      const seen = new Set<number>()
      const queue: number[] = [rootPid]

      while (queue.length > 0) {
        const pid = queue.shift()!
        if (seen.has(pid)) continue
        seen.add(pid)

        const row = table.get(pid)
        if (!row) continue

        cpu += row.cpu
        rss += row.rss

        const kids = children.get(pid)
        if (kids) queue.push(...kids)
      }

      samples.push({ pid: rootPid, cpuPercent: cpu, rssBytes: rss, timestamp })
    }

    return samples
  })
}
