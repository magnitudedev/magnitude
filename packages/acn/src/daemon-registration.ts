import * as FileSystem from "@effect/platform/FileSystem";
import type { PlatformError } from "@effect/platform/Error";
import * as NodePath from "path";
import { Effect, Option } from "effect";
import {
  readStructuredFile,
  StructuredFileEncodeFailed,
  writeStructuredFileAtomic,
} from "@magnitudedev/storage";
import {
  AcnVersionRegistryOwnershipSchema,
  AcnVersionRegistrySchema,
  AcnInstanceRecordSchema,
  type AcnInstanceRecord,
  type AcnRegistration,
  type AcnRegistrationOwnership,
  type AcnVersionRegistry,
} from "@magnitudedev/acn-protocol";

export type { AcnRegistration, AcnVersionRegistry };

export interface RegisteredAcnInstance {
  readonly path: string;
  readonly record: AcnInstanceRecord;
}

export const registrationPath = (dataDir: string): string =>
  NodePath.join(dataDir, "acn", "registry.json");

export const instancesDirectory = (dataDir: string): string =>
  NodePath.join(dataDir, "acn", "instances");

export const instancePath = (dataDir: string, id: string): string =>
  NodePath.join(instancesDirectory(dataDir), `${encodeURIComponent(id)}.json`);

export const registrationIsOwnedBy = (
  registration: Option.Option<AcnRegistrationOwnership>,
  ownerId: string
): boolean => Option.exists(registration, (value) => value.id === ownerId);

export const readRegistrationOwnership = (
  path: string
): Effect.Effect<
  Option.Option<AcnRegistrationOwnership>,
  PlatformError,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const result = yield* readStructuredFile(
      path,
      AcnVersionRegistryOwnershipSchema
    ).pipe(Effect.provideService(FileSystem.FileSystem, fs));
    if (result._tag !== "Present") return Option.none();
    return result.value.registration;
  });

export const readRegistration = (
  path: string
): Effect.Effect<
  Option.Option<AcnRegistration>,
  PlatformError,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const result = yield* readStructuredFile(
      path,
      AcnVersionRegistrySchema
    ).pipe(Effect.provideService(FileSystem.FileSystem, fs));
    if (result._tag === "Missing") return Option.none();
    if (result._tag === "Invalid") {
      yield* Effect.logError("Failed to parse ACN registration file").pipe(
        Effect.annotateLogs({ path, error: result.error.reason })
      );
      return Option.none();
    }
    return result.value.registration;
  });

export const writeRegistrationAtomic = (
  path: string,
  registration: AcnRegistration
): Effect.Effect<
  void,
  PlatformError | StructuredFileEncodeFailed,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const directory = NodePath.dirname(path);
    yield* fs.makeDirectory(directory, { recursive: true });
    yield* fs.chmod(directory, 0o700);
    const registry: AcnVersionRegistry = {
      schemaVersion: 1,
      registration: Option.some(registration),
    };

    yield* writeStructuredFileAtomic(path, AcnVersionRegistrySchema, registry, {
      mode: 0o600,
    });
  });

export const writeInstanceAtomic = (
  dataDir: string,
  record: AcnInstanceRecord,
): Effect.Effect<
  string,
  PlatformError | StructuredFileEncodeFailed,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const directory = instancesDirectory(dataDir);
    yield* fs.makeDirectory(directory, { recursive: true });
    yield* fs.chmod(directory, 0o700);
    const path = instancePath(dataDir, record.id);
    yield* writeStructuredFileAtomic(path, AcnInstanceRecordSchema, record, {
      mode: 0o600,
    });
    return path;
  });

export const removeExactInstance = (
  instance: RegisteredAcnInstance,
): Effect.Effect<void, PlatformError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const current = yield* readStructuredFile(
      instance.path,
      AcnInstanceRecordSchema,
    );
    if (
      current._tag === "Present" &&
      current.value.id === instance.record.id &&
      current.value.pid === instance.record.pid &&
      current.value.processStartIdentity === instance.record.processStartIdentity
    ) {
      const fs = yield* FileSystem.FileSystem;
      yield* fs.remove(instance.path, { force: true });
    }
  });

export const listAcnInstances = (
  dataDir: string,
): Effect.Effect<
  readonly RegisteredAcnInstance[],
  PlatformError,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const directory = instancesDirectory(dataDir);
    const names = yield* fs.readDirectory(directory).pipe(
      Effect.catchAll((error) =>
        error._tag === "SystemError" && error.reason === "NotFound"
          ? Effect.succeed([])
          : Effect.fail(error),
      ),
    );
    const records = yield* Effect.forEach(
      names.filter((name) => name.endsWith(".json")),
      (name) => {
        const path = NodePath.join(directory, name);
        return readStructuredFile(path, AcnInstanceRecordSchema).pipe(
          Effect.flatMap((result) => {
            if (result._tag === "Present") {
              return Effect.succeed(Option.some({ path, record: result.value }));
            }
            return fs.remove(path, { force: true }).pipe(
              Effect.as(Option.none<RegisteredAcnInstance>()),
            );
          }),
        );
      },
    );
    return records.filter(Option.isSome).map(({ value }) => value);
  });
