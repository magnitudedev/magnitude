import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Context, Effect, Layer, type Option } from "effect"
import type { AcnRevision } from "../acn-revision"
import { makeAcnCoordinationDatabase } from "./coordination-database"
import type { AcnProcessStoreError } from "./errors"
import { SqliteDriver } from "./sqlite-driver"

export interface AcnRevisionStore {
  readonly register: (revision: AcnRevision) => Effect.Effect<void, AcnProcessStoreError>
  readonly selected: Effect.Effect<Option.Option<AcnRevision>, AcnProcessStoreError>
}

export const AcnRevisionStore = Context.GenericTag<AcnRevisionStore>(
  "@magnitudedev/acn-protocol/coordination/AcnRevisionStore",
)

export const makeAcnRevisionStore = (
  dataDirectory: string,
): Effect.Effect<
  AcnRevisionStore,
  never,
  FileSystem.FileSystem | Path.Path | SqliteDriver
> => makeAcnCoordinationDatabase(dataDirectory).pipe(
  Effect.map((database) => AcnRevisionStore.of({
    register: database.registerRevision,
    selected: database.selectedRevision,
  })),
)

export const AcnRevisionStoreLive = (dataDirectory: string) =>
  Layer.effect(AcnRevisionStore, makeAcnRevisionStore(dataDirectory))
