import * as FileSystem from "@effect/platform/FileSystem"
import * as NodePath from "path"
import { Console, Data, Effect } from "effect"
import { defaultDataDir } from "./daemon-lifecycle"
import { listRegisteredAcns, type RegisteredAcn } from "./daemon-registration"

type KillResult =
  | { readonly _tag: "killed"; readonly acn: RegisteredAcn }
  | { readonly _tag: "stale"; readonly acn: RegisteredAcn }
  | { readonly _tag: "skippedSelf"; readonly acn: RegisteredAcn }
  | { readonly _tag: "failed"; readonly acn: RegisteredAcn; readonly error: ProcessKillError }

class ProcessKillError extends Data.TaggedError("ProcessKillError")<{
  readonly cause: unknown
  readonly code?: string
  readonly message: string
}> {}

const errorCode = (error: unknown): string | undefined => {
  if (typeof error !== "object" || error === null || !("code" in error)) return undefined
  const code = Reflect.get(error, "code")
  return typeof code === "string" ? code : undefined
}

const toProcessKillError = (error: unknown): ProcessKillError =>
  new ProcessKillError({
    cause: error,
    code: errorCode(error),
    message: error instanceof Error ? error.message : String(error),
  })

const isMissingProcess = (error: ProcessKillError): boolean =>
  error.code === "ESRCH"

const removeStaleRegistration = (path: string) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    yield* fs.remove(path, { force: true })
    yield* fs.remove(NodePath.dirname(path), { recursive: false }).pipe(
      Effect.catchAll(() => Effect.void),
    )
  })

const killRegisteredAcn = (acn: RegisteredAcn): Effect.Effect<KillResult, never, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    if (acn.registration.pid === process.pid) {
      return { _tag: "skippedSelf", acn } satisfies KillResult
    }

    const killed = yield* Effect.try({
      try: () => process.kill(acn.registration.pid, "SIGTERM"),
      catch: toProcessKillError,
    }).pipe(Effect.either)

    if (killed._tag === "Right") {
      return { _tag: "killed", acn } satisfies KillResult
    }

    if (isMissingProcess(killed.left)) {
      yield* removeStaleRegistration(acn.path).pipe(Effect.catchAll(() => Effect.void))
      return { _tag: "stale", acn } satisfies KillResult
    }

    return { _tag: "failed", acn, error: killed.left } satisfies KillResult
  })

const resultLine = (result: KillResult): string => {
  const { registration } = result.acn
  const label = `${registration.version} pid ${registration.pid}`
  switch (result._tag) {
    case "killed":
      return `killed ${label}`
    case "stale":
      return `removed stale registration ${label}`
    case "skippedSelf":
      return `skipped current process ${label}`
    case "failed":
      return `failed ${label}: ${result.error.message}`
  }
}

export const killAllAcns = Effect.gen(function* () {
  const acns = yield* listRegisteredAcns(defaultDataDir())
  if (acns.length === 0) {
    yield* Console.log("No registered ACNs found.")
    return
  }

  const results = yield* Effect.forEach(acns, killRegisteredAcn)
  for (const result of results) {
    yield* Console.log(resultLine(result))
  }
})
