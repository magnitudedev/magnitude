import { Effect, Ref } from "effect"
import type { DownloadModelParams, DownloadState } from "./types"

/**
 * Internal download state tracker.
 * Not a public API — the store owns this and updates it as downloads progress.
 */
export interface DownloadRegistry {
  readonly register: (params: DownloadModelParams, totalBytes: number) => Effect.Effect<string>
  readonly updateProgress: (id: string, progress: Partial<DownloadState>) => Effect.Effect<void>
  readonly markCompleted: (id: string) => Effect.Effect<void>
  readonly markFailed: (id: string, error: string) => Effect.Effect<void>
  readonly markPaused: (id: string) => Effect.Effect<void>
  readonly remove: (id: string) => Effect.Effect<void>
  readonly list: () => Effect.Effect<readonly DownloadState[]>
  readonly get: (id: string) => Effect.Effect<DownloadState | null>
}

export function makeDownloadRegistry(): DownloadRegistry {
  const ref = Effect.runSync(Ref.make<Map<string, DownloadState>>(new Map()))

  const register: DownloadRegistry["register"] = (params, totalBytes) =>
    Effect.gen(function* () {
      const id = `${params.repo}/${params.file}`
      const state: DownloadState = {
        id,
        repo: params.repo,
        file: params.file,
        status: "downloading",
        downloadedBytes: 0,
        totalBytes,
        percent: 0,
        bytesPerSecond: 0,
        etaSeconds: 0,
      }
      yield* Ref.update(ref, (map) => new Map(map).set(id, state))
      return id
    })

  const updateProgress: DownloadRegistry["updateProgress"] = (id, progress) =>
    Ref.update(ref, (map) => {
      const current = map.get(id)
      if (!current) return map
      return new Map(map).set(id, { ...current, ...progress })
    })

  const markCompleted: DownloadRegistry["markCompleted"] = (id) =>
    Ref.update(ref, (map) => {
      const current = map.get(id)
      if (!current) return map
      return new Map(map).set(id, {
        ...current,
        status: "completed",
        percent: 100,
      })
    })

  const markFailed: DownloadRegistry["markFailed"] = (id, error) =>
    Ref.update(ref, (map) => {
      const current = map.get(id)
      if (!current) return map
      return new Map(map).set(id, { ...current, status: "failed", error })
    })

  const markPaused: DownloadRegistry["markPaused"] = (id) =>
    Ref.update(ref, (map) => {
      const current = map.get(id)
      if (!current) return map
      return new Map(map).set(id, { ...current, status: "paused" })
    })

  const remove: DownloadRegistry["remove"] = (id) =>
    Ref.update(ref, (map) => {
      const next = new Map(map)
      next.delete(id)
      return next
    })

  const list: DownloadRegistry["list"] = () =>
    Effect.gen(function* () {
      const map = yield* Ref.get(ref)
      return Array.from(map.values())
    })

  const get: DownloadRegistry["get"] = (id) =>
    Effect.gen(function* () {
      const map = yield* Ref.get(ref)
      return map.get(id) ?? null
    })

  return { register, updateProgress, markCompleted, markFailed, markPaused, remove, list, get }
}
