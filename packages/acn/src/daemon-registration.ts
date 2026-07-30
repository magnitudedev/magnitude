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
  type AcnRegistration,
  type AcnRegistrationOwnership,
  type AcnVersionRegistry,
} from "@magnitudedev/acn-protocol";

export type { AcnRegistration, AcnVersionRegistry };

export interface RegisteredAcn {
  readonly path: string;
  readonly registration: AcnRegistration;
}

export const registrationPath = (dataDir: string): string =>
  NodePath.join(dataDir, "acn", "registry.json");

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
    const registry: AcnVersionRegistry = {
      schemaVersion: 1,
      registration: Option.some(registration),
    };

    yield* writeStructuredFileAtomic(path, AcnVersionRegistrySchema, registry, {
      mode: 0o600,
    });
  });

export const listRegisteredAcns = (
  dataDir: string
): Effect.Effect<
  readonly RegisteredAcn[],
  PlatformError,
  FileSystem.FileSystem
> =>
  Effect.gen(function* () {
    const path = registrationPath(dataDir);
    const registration = yield* readRegistration(path);
    return Option.match(registration, {
      onNone: () => [],
      onSome: (value) => [{ path, registration: value }],
    });
  });
