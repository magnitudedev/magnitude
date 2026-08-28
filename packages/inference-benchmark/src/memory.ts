import { dlopen, FFIType, ptr } from "bun:ffi"
import { readFile } from "node:fs/promises"
import { Effect, Fiber, Ref } from "effect"
import type { MemoryObservation, MemorySample } from "./domain"

interface ProcessRow {
  readonly pid: number
  readonly ppid: number
  readonly rssBytes: number
}

function processRows(): readonly ProcessRow[] {
  const result = Bun.spawnSync(["ps", "-axo", "pid=,ppid=,rss="])
  if (result.exitCode !== 0) return []
  return result.stdout.toString().split(/\r?\n/).flatMap((line) => {
    const [pidRaw, ppidRaw, rssRaw] = line.trim().split(/\s+/)
    const pid = Number(pidRaw)
    const ppid = Number(ppidRaw)
    const rssKiB = Number(rssRaw)
    return Number.isInteger(pid) && Number.isInteger(ppid) && Number.isFinite(rssKiB)
      ? [{ pid, ppid, rssBytes: rssKiB * 1024 }]
      : []
  })
}

function processTree(rootPid: number, rows: readonly ProcessRow[]): readonly ProcessRow[] {
  const selected = new Map<number, ProcessRow>()
  const queue = [rootPid]
  while (queue.length > 0) {
    const pid = queue.shift()!
    const row = rows.find((candidate) => candidate.pid === pid)
    if (row) selected.set(pid, row)
    for (const child of rows) if (child.ppid === pid && !selected.has(child.pid)) queue.push(child.pid)
  }
  return [...selected.values()]
}

export function processTreePids(rootPid: number): readonly number[] {
  if (!Number.isSafeInteger(rootPid) || rootPid <= 1) return []
  return processTree(rootPid, processRows()).map(({ pid }) => pid)
}

const libproc = process.platform === "darwin"
  ? dlopen("/usr/lib/libproc.dylib", {
      proc_pid_rusage: { args: [FFIType.i32, FFIType.i32, FFIType.ptr], returns: FFIType.i32 },
    } as const)
  : undefined

function macPhysicalFootprint(pid: number): number | undefined {
  if (process.platform !== "darwin") return undefined
  try {
    if (!libproc) return undefined
    const buffer = new Uint8Array(512)
    const result = libproc.symbols.proc_pid_rusage(pid, 4, ptr(buffer))
    if (result !== 0) return undefined
    return Number(new DataView(buffer.buffer).getBigUint64(72, true))
  } catch {
    return undefined
  }
}

const linuxPss = (pid: number): Effect.Effect<number | undefined> =>
  Effect.tryPromise(() => readFile(`/proc/${pid}/smaps_rollup`, "utf8")).pipe(
    Effect.map((text) => {
      const match = /^Pss:\s+(\d+)\s+kB$/m.exec(text)
      return match ? Number(match[1]) * 1024 : undefined
    }),
    Effect.catchAll(() => Effect.sync((): number | undefined => undefined)),
  )

function nvidiaProcessMemory(pids: ReadonlySet<number>): Readonly<Record<string, number>> | undefined {
  if (process.platform !== "linux" || pids.size === 0) return undefined
  const result = Bun.spawnSync([
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid,used_memory",
    "--format=csv,noheader,nounits",
  ], { stderr: "ignore" })
  if (result.exitCode !== 0) return undefined
  const totals: Record<string, number> = {}
  for (const line of result.stdout.toString().split(/\r?\n/)) {
    const [pidRaw, uuidRaw, mibRaw] = line.split(",").map((part) => part.trim())
    const pid = Number(pidRaw)
    const mib = Number(mibRaw)
    if (!pids.has(pid) || !uuidRaw || !Number.isFinite(mib)) continue
    totals[uuidRaw] = (totals[uuidRaw] ?? 0) + mib * 1024 * 1024
  }
  return Object.keys(totals).length > 0 ? totals : undefined
}

export interface MemoryProbeOptions {
  readonly rootPid: number
  readonly intervalMs?: number
}

export interface ActiveMemoryProbe {
  readonly baseline: MemorySample
  readonly samples: Effect.Effect<readonly MemorySample[]>
  readonly stop: Effect.Effect<MemoryObservation>
}

const takeSample = (rootPid: number, started: number, includeDevice = true): Effect.Effect<MemorySample> => Effect.gen(function* () {
  const tree = processTree(rootPid, processRows())
  let hostBytes: number | undefined
  if (process.platform === "darwin") {
    const footprints = tree.map((row) => macPhysicalFootprint(row.pid)).filter((value): value is number => value !== undefined)
    if (footprints.length > 0) hostBytes = footprints.reduce((sum, value) => sum + value, 0)
  } else if (process.platform === "linux") {
    const pss = (yield* Effect.forEach(tree, (row) => linuxPss(row.pid), { concurrency: "unbounded" }))
      .filter((value): value is number => value !== undefined)
    hostBytes = pss.length > 0 ? pss.reduce((sum, value) => sum + value, 0) : tree.reduce((sum, row) => sum + row.rssBytes, 0)
  } else {
    hostBytes = tree.length > 0 ? tree.reduce((sum, row) => sum + row.rssBytes, 0) : undefined
  }
  return {
    atMs: performance.now() - started,
    hostBytes,
    deviceBytes: includeDevice ? nvidiaProcessMemory(new Set(tree.map((row) => row.pid))) : undefined,
  }
})

export const startMemoryProbe = (options: MemoryProbeOptions): Effect.Effect<ActiveMemoryProbe> => Effect.gen(function* () {
  const started = performance.now()
  const baseline = yield* takeSample(options.rootPid, started)
  const samples = yield* Ref.make<readonly MemorySample[]>([baseline])
  const sampleIndex = yield* Ref.make(0)
  const sampler = Effect.gen(function* () {
    yield* Effect.sleep(`${options.intervalMs ?? 250} millis`)
    const index = yield* Ref.updateAndGet(sampleIndex, (value) => value + 1)
    const sample = yield* takeSample(options.rootPid, started, index % 4 === 0)
    yield* Ref.update(samples, (current) => [...current, sample])
  }).pipe(Effect.forever, Effect.fork)
  const fiber = yield* sampler

  return {
    baseline,
    samples: Ref.get(samples),
    stop: Effect.gen(function* () {
      yield* Fiber.interrupt(fiber)
      yield* Effect.sleep("50 millis")
      const finalSample = yield* takeSample(options.rootPid, started)
      const collected = yield* Ref.updateAndGet(samples, (current) => [...current, finalSample])
      const host = collected.map((sample) => sample.hostBytes).filter((value): value is number => value !== undefined)
      const deviceIds = new Set(collected.flatMap((sample) => Object.keys(sample.deviceBytes ?? {})))
      const peakDeviceBytes = deviceIds.size > 0 ? Object.fromEntries([...deviceIds].map((id) => [
        id,
        Math.max(...collected.map((sample) => sample.deviceBytes?.[id] ?? 0)),
      ])) : undefined
      const supported = host.length > 0
      return {
        supported,
        source: process.platform === "darwin"
          ? "mach-proc-pid-rusage"
          : process.platform === "linux" && deviceIds.size > 0 ? "procfs-pss+nvml-process" : process.platform === "linux" ? "procfs-pss" : "process-rss",
        scope: `process-tree:${options.rootPid}`,
        baselineBytes: baseline.hostBytes,
        peakBytes: supported ? Math.max(...host) : undefined,
        retainedBytes: collected.at(-1)?.hostBytes,
        baselineDeviceBytes: baseline.deviceBytes,
        peakDeviceBytes,
        retainedDeviceBytes: collected.at(-1)?.deviceBytes,
        samples: collected,
        limitation: process.platform === "darwin"
          ? "Mach physical footprint for the process tree; excludes file-backed model mappings and cannot separate Metal, CPU, or KV allocations"
          : undefined,
      }
    }),
  }
})
