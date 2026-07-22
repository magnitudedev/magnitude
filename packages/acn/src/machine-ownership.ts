import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Schema, Scope } from "effect"
import * as NodePath from "node:path"

const OwnerSchema = Schema.Struct({
  id: Schema.String,
  pid: Schema.Number,
  version: Schema.String,
  startedAt: Schema.Number,
})

type Owner = typeof OwnerSchema.Type

export class AcnMachineAlreadyOwned extends Data.TaggedError("AcnMachineAlreadyOwned")<{
  readonly owner: Owner
}> {}

export class AcnMachineOwnershipFailed extends Data.TaggedError("AcnMachineOwnershipFailed")<{
  readonly operation: string
  readonly cause: unknown
}> {}

const reason = (cause: unknown): string | undefined =>
  typeof cause === "object" && cause !== null && "reason" in cause
    ? String(Reflect.get(cause, "reason"))
    : undefined

const isAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0)
    return true
  } catch (cause) {
    return !(cause instanceof Error && "code" in cause && cause.code === "ESRCH")
  }
}

const failure = (operation: string) => (cause: unknown) =>
  new AcnMachineOwnershipFailed({ operation, cause })

/**
 * Acquires the machine-wide ACN ownership record before any server or ICN
 * resources are constructed. The exact hard-linked owner record makes stale
 * recovery safe against delayed contenders.
 */
export const acquireAcnMachineOwnership = (input: {
  readonly dataDir: string
  readonly id: string
  readonly version: string
}): Effect.Effect<
  void,
  AcnMachineAlreadyOwned | AcnMachineOwnershipFailed,
  FileSystem.FileSystem | Scope.Scope
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const directory = NodePath.join(input.dataDir, "acn")
    const path = NodePath.join(directory, "owner")
    const owner: Owner = {
      id: input.id,
      pid: process.pid,
      version: input.version,
      startedAt: Date.now(),
    }
    const encoded = yield* Schema.encode(Schema.parseJson(OwnerSchema))(owner).pipe(
      Effect.mapError(failure("encode owner record")),
    )

    const readOwner = (target: string) =>
      fs.readFileString(target).pipe(
        Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(OwnerSchema))),
        Effect.map((value) => ({ raw: JSON.stringify(value), value })),
        Effect.catchAll(() => Effect.succeed(null)),
      )

    const acquire: Effect.Effect<
      void,
      AcnMachineAlreadyOwned | AcnMachineOwnershipFailed
    > = Effect.suspend(() =>
      Effect.gen(function* () {
        yield* fs.makeDirectory(directory, { recursive: true }).pipe(
          Effect.mapError(failure("create ACN ownership directory")),
        )
        yield* fs.chmod(directory, 0o700).pipe(Effect.ignore)
        const publication = `${path}.publishing-${encodeURIComponent(input.id)}`
        yield* fs.writeFileString(publication, encoded, { mode: 0o600 }).pipe(
          Effect.mapError(failure("write ACN owner publication")),
        )
        const linked = yield* fs.link(publication, path).pipe(
          Effect.as(true),
          Effect.catchAll((cause) =>
            reason(cause) === "AlreadyExists"
              ? Effect.succeed(false)
              : Effect.fail(failure("publish ACN ownership")(cause)),
          ),
          Effect.ensuring(fs.remove(publication, { force: true }).pipe(Effect.ignore)),
        )
        if (linked) return

        const observed = yield* readOwner(path)
        if (observed && isAlive(observed.value.pid)) {
          return yield* new AcnMachineAlreadyOwned({ owner: observed.value })
        }

        // Preserve an exact hard link before removal. A delayed contender can
        // only remove the owner bytes it actually observed, never a successor.
        const tombstone = `${path}.stale-${encodeURIComponent(crypto.randomUUID())}`
        yield* fs.link(path, tombstone).pipe(
          Effect.catchAll((cause) =>
            reason(cause) === "AlreadyExists" || reason(cause) === "NotFound"
              ? Effect.void
              : Effect.fail(failure("quarantine stale ACN owner")(cause)),
          ),
        )
        const [currentRaw, staleRaw] = yield* Effect.all([
          fs.readFileString(path).pipe(Effect.option),
          fs.readFileString(tombstone).pipe(Effect.option),
        ])
        if (
          currentRaw._tag === "Some" &&
          staleRaw._tag === "Some" &&
          currentRaw.value === staleRaw.value
        ) {
          yield* fs.remove(path, { force: true }).pipe(
            Effect.mapError(failure("remove stale ACN owner")),
          )
        }
        yield* Effect.sleep("25 millis")
        return yield* acquire
      }),
    )

    yield* Effect.acquireRelease(acquire, () =>
      fs.readFileString(path).pipe(
        Effect.flatMap((current) =>
          current === encoded ? fs.remove(path, { force: true }) : Effect.void,
        ),
        Effect.catchAll(() => Effect.void),
      ),
    )
  })
