import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Context, Effect, Layer, type Option } from "effect"
import {
  makeAcnCoordinationDatabase,
  type ReplaceOwnerResult,
} from "./coordination-database"
import type { AcnProcessStoreError } from "./errors"
import type { AcnOwnerRecord } from "./schemas"
import { SqliteDriver } from "./sqlite-driver"

export interface AcnOwnerStore {
  readonly current: Effect.Effect<Option.Option<AcnOwnerRecord>, AcnProcessStoreError>
  readonly replaceOwner: (
    expectedOwner: Option.Option<AcnOwnerRecord>,
    candidateOwner: AcnOwnerRecord,
  ) => Effect.Effect<ReplaceOwnerResult, AcnProcessStoreError>
}

export const AcnOwnerStore = Context.GenericTag<AcnOwnerStore>(
  "@magnitudedev/acn-protocol/coordination/AcnOwnerStore",
)

export const makeAcnOwnerStore = (
  dataDirectory: string,
): Effect.Effect<
  AcnOwnerStore,
  never,
  FileSystem.FileSystem | Path.Path | SqliteDriver
> => makeAcnCoordinationDatabase(dataDirectory).pipe(
  Effect.map((database) => AcnOwnerStore.of({
    current: database.currentOwner,
    replaceOwner: database.replaceOwner,
  })),
)

export const AcnOwnerStoreLive = (dataDirectory: string) =>
  Layer.effect(AcnOwnerStore, makeAcnOwnerStore(dataDirectory))
