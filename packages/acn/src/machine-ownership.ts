import * as FileSystem from "@effect/platform/FileSystem"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import type { PlatformError } from "@effect/platform/Error"
import { Data, Effect, Option, Schema, Scope } from "effect"
import type * as ParseResult from "effect/ParseResult"
import * as NodePath from "node:path"
import {
  AcnOwnerIdSchema,
  AcnProcessStartIdentitySchema,
  type AcnOwnerId,
} from "@magnitudedev/acn-protocol"
import {
  currentProcessStartIdentity,
  readProcessStartIdentity,
} from "./process-identity"

const OwnerSchema = Schema.Struct({
  id: AcnOwnerIdSchema,
  pid: Schema.Number,
  version: Schema.String,
  startedAt: Schema.Number,
  processStartIdentity: AcnProcessStartIdentitySchema,
})

type Owner = typeof OwnerSchema.Type

export class AcnMachineOwnershipFailed extends Data.TaggedError("AcnMachineOwnershipFailed")<{
  readonly operation: string
  readonly reason: string
}> {}

const platformErrorReason = (cause: PlatformError): Option.Option<string> =>
  cause._tag === "SystemError" ? Option.some(cause.reason) : Option.none()

const hasPlatformErrorReason = (
  cause: PlatformError,
  expected: string,
): boolean =>
  Option.exists(platformErrorReason(cause), (reason) => reason === expected)

const failure =
  (operation: string) =>
  (cause: PlatformError | ParseResult.ParseError) =>
    new AcnMachineOwnershipFailed({ operation, reason: String(cause) })

/**
 * Waits for and acquires the machine-wide active-runtime ownership record.
 * The canonical ACN server is already registered while this waits, allowing
 * the predecessor to observe ownership loss and retire itself. Exact
 * hard-linked records make stale recovery safe against delayed contenders.
 */
export const acquireAcnMachineOwnership = (input: {
  readonly dataDir: string
  readonly id: AcnOwnerId
  readonly version: string
}): Effect.Effect<
  void,
  AcnMachineOwnershipFailed,
  FileSystem.FileSystem | CommandExecutor.CommandExecutor | Scope.Scope
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const directory = NodePath.join(input.dataDir, "acn")
    const path = NodePath.join(directory, "owner")
    const processStartIdentity = yield* currentProcessStartIdentity.pipe(
      Effect.mapError((cause) =>
        new AcnMachineOwnershipFailed({
          operation: "inspect current ACN process",
          reason: cause.reason,
        }),
      ),
    )
    const owner: Owner = {
      id: input.id,
      pid: process.pid,
      version: input.version,
      startedAt: Date.now(),
      processStartIdentity,
    }
    const encoded = yield* Schema.encode(Schema.parseJson(OwnerSchema))(owner).pipe(
      Effect.mapError(failure("encode owner record")),
    )

    const readOwner = (target: string) =>
      fs.readFileString(target).pipe(
        Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(OwnerSchema))),
        Effect.option,
      )

    const acquire: Effect.Effect<
      void,
      AcnMachineOwnershipFailed,
      CommandExecutor.CommandExecutor
    > = Effect.suspend(() =>
      Effect.gen(function* () {
        yield* fs.makeDirectory(directory, { recursive: true }).pipe(
          Effect.mapError(failure("create ACN ownership directory")),
        )
        yield* fs.chmod(directory, 0o700).pipe(
          Effect.mapError(failure("secure ACN ownership directory")),
        )
        const publication = `${path}.publishing-${encodeURIComponent(input.id)}`
        yield* fs.writeFileString(publication, encoded, { mode: 0o600 }).pipe(
          Effect.mapError(failure("write ACN owner publication")),
        )
        const linked = yield* fs.link(publication, path).pipe(
          Effect.as(true),
          Effect.catchAll((cause) =>
            hasPlatformErrorReason(cause, "AlreadyExists")
              ? Effect.succeed(false)
              : Effect.fail(failure("publish ACN ownership")(cause)),
          ),
          Effect.ensuring(fs.remove(publication, { force: true }).pipe(Effect.ignore)),
        )
        if (linked) return

        const observed = yield* readOwner(path)
        const observedIsAlive = yield* Option.match(observed, {
          onNone: () => Effect.succeed(false),
          onSome: (owner) => readProcessStartIdentity(owner.pid).pipe(
            Effect.mapError((cause) =>
              new AcnMachineOwnershipFailed({
                operation: "inspect incumbent ACN process",
                reason: cause.reason,
              }),
            ),
            Effect.map((identity) => Option.exists(
              identity,
              (value) => value === owner.processStartIdentity,
            )),
          ),
        })
        if (observedIsAlive) {
          yield* Effect.sleep("100 millis")
          return yield* acquire
        }

        // Preserve an exact hard link before removal. A delayed contender can
        // only remove the owner bytes it actually observed, never a successor.
        const tombstone = `${path}.stale-${encodeURIComponent(crypto.randomUUID())}`
        yield* fs.link(path, tombstone).pipe(
          Effect.catchAll((cause) =>
            hasPlatformErrorReason(cause, "AlreadyExists") ||
            hasPlatformErrorReason(cause, "NotFound")
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

    yield* Effect.acquireReleaseInterruptible(acquire, () =>
      fs.readFileString(path).pipe(
        Effect.flatMap((current) =>
          current === encoded ? fs.remove(path, { force: true }) : Effect.void,
        ),
        Effect.catchAll(() => Effect.void),
      ),
    )
  })
