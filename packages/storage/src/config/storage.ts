import * as FileSystem from "@effect/platform/FileSystem";
import * as Path from "@effect/platform/Path";
import { type PlatformError } from "@effect/platform/Error";
import { randomUUID } from "node:crypto";
import { Effect, Option, Schema } from "effect";

import {
  makeStorageIo,
  type JsonError,
  SchemaDecodeError,
  SchemaEncodeError,
} from "../io/storage";
import {
  readRecoverableStructuredFile,
  writeStructuredFileAtomic,
  writeTextFileAtomic,
} from "../io/structured-file";
import { GlobalStorage } from "../services";
import {
  MagnitudeConfigSchema,
  resolveContextLimitPolicy,
  type ContextLimitPolicy,
  type MagnitudeConfig,
} from "../types/config";
import type { ConfigStorageShape } from "./contracts";

const DEFAULT_CONFIG = Schema.decodeUnknownSync(MagnitudeConfigSchema)({});

const discardRemovedModelConfiguration = (config: MagnitudeConfig): {
  readonly value: MagnitudeConfig;
  readonly changed: boolean;
} => {
  const models = config.models === undefined ? undefined : {
    slots: config.models.slots,
    localModelRecency: config.models.localModelRecency,
  };
  const removedModelFields = config.models !== undefined
    && Reflect.ownKeys(config.models).some((key) => key !== "slots" && key !== "localModelRecency");
  const value = models === undefined ? { ...config } : { ...config, models };
  const removedLocalInference = Reflect.has(value, "localInference");
  if (removedLocalInference) Reflect.deleteProperty(value, "localInference");
  return { value, changed: removedModelFields || removedLocalInference };
};

const safeRecoveryMessage = (message: string): string =>
  message.replace(/, actual[\s\S]*$/, "").slice(0, 500);

export function makeConfigStorage(): Effect.Effect<
  ConfigStorageShape,
  never,
  FileSystem.FileSystem | Path.Path | GlobalStorage
> {
  return Effect.gen(function* () {
    const io = yield* makeStorageIo();
    const fs = yield* FileSystem.FileSystem;
    const globalStorage = yield* GlobalStorage;
    const g = globalStorage.paths;

    const writeConfigUnlocked = (
      config: MagnitudeConfig
    ): Effect.Effect<void, PlatformError | JsonError> =>
      writeStructuredFileAtomic(g.configFile, MagnitudeConfigSchema, config, {
        parseOptions: { onExcessProperty: "preserve" },
      }).pipe(
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.mapError((error) =>
          error._tag === "StructuredFileEncodeFailed"
            ? new SchemaEncodeError({ path: g.configFile, message: error.reason })
            : error
        )
      );

    const corruptBackupPath = (): string => {
      const timestamp = new Date().toISOString().replaceAll(/[-:.]/g, "");
      return `${g.configFile}.corrupt-${timestamp}-${randomUUID()}`;
    };

    const preserveCorruptOriginal = (
      originalText: string
    ): Effect.Effect<string, PlatformError> =>
      Effect.gen(function* () {
        const backupPath = corruptBackupPath();
        yield* writeTextFileAtomic(backupPath, originalText).pipe(
          Effect.provideService(FileSystem.FileSystem, fs)
        );
        return backupPath;
      });

    const readConfigUnlocked = (): Effect.Effect<MagnitudeConfig, PlatformError | JsonError> =>
      Effect.gen(function* () {
        const result = yield* readRecoverableStructuredFile(
          g.configFile,
          MagnitudeConfigSchema,
          { rootDefault: () => DEFAULT_CONFIG }
        ).pipe(
          Effect.provideService(FileSystem.FileSystem, fs)
        );
        if (result._tag === "Missing") return DEFAULT_CONFIG;
        if (result._tag === "Unrecoverable") {
          return yield* new SchemaDecodeError({ path: g.configFile, message: result.reason });
        }
        if (result._tag === "Malformed") {
          const backupPath = yield* preserveCorruptOriginal(result.originalText);
          yield* writeConfigUnlocked(DEFAULT_CONFIG);
          yield* Effect.logWarning("Recovered malformed Magnitude config").pipe(
            Effect.annotateLogs({
              path: g.configFile,
              backupPath,
              reason: result.reason.slice(0, 1_000),
            })
          );
          return DEFAULT_CONFIG;
        }
        const cleaned = discardRemovedModelConfiguration(result.value);
        const backupPath = result.recovery.recovered && result.recovery.resetRoot
          ? yield* preserveCorruptOriginal(result.originalText)
          : undefined;
        if (result.recovery.recovered || cleaned.changed) {
          yield* writeConfigUnlocked(cleaned.value);
        }
        if (result.recovery.recovered) {
          yield* Effect.logWarning("Recovered invalid Magnitude config values").pipe(
            Effect.annotateLogs({
              path: g.configFile,
              resetRoot: result.recovery.resetRoot,
              attempts: result.recovery.attempts,
              removedPaths: result.recovery.removedPaths
                .map((parts) => parts.map(String).join("."))
                .join(","),
              issues: result.recovery.issues
                .map((issue) =>
                  `${issue.path.map(String).join(".")}: ${safeRecoveryMessage(issue.message)}`
                )
                .join(" | ")
                .slice(0, 4_000),
              ...(backupPath ? { backupPath } : {}),
            })
          );
        }
        return cleaned.value;
      });

    const readConfig = (): Effect.Effect<MagnitudeConfig, PlatformError | JsonError> =>
      io.withPathLock(g.configFile, readConfigUnlocked());

    return {
      load: () => readConfig(),

      save: (config) =>
        io.withPathLock(g.configFile, writeConfigUnlocked(config)),

      update: (f) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfigUnlocked();
            const next = f(current);
            yield* writeConfigUnlocked(next);
            return next;
          })
        ),

      getContextLimitPolicy: () =>
        readConfig().pipe(Effect.map(resolveContextLimitPolicy)),

      setContextLimitPolicy: (policy) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfigUnlocked();
            yield* writeConfigUnlocked({
              ...current,
              contextLimits: {
                ...(current.contextLimits ?? {}),
                ...policy,
              },
            });
          })
        ),

      updateModelSlot: (slotId, selection) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfigUnlocked();
            const existingModels = current.models ?? {
              slots: { primary: Option.none(), secondary: Option.none() },
              localModelRecency: { primary: [], secondary: [] },
            };
            yield* writeConfigUnlocked({
              ...current,
              models: {
                ...existingModels,
                slots: { ...existingModels.slots, [slotId]: selection },
              },
            });
          })
        ),

      getOnboardingConfig: () =>
        readConfig().pipe(Effect.map((config) => config.onboarding ?? null)),


      completeOnboardingFlow: (flowId, version, completedAt) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfigUnlocked();
            yield* writeConfigUnlocked({
              ...current,
              onboarding: {
                ...current.onboarding,
                completions: {
                  ...current.onboarding?.completions,
                  [flowId]: { version, completedAt },
                },
              },
            });
          })
        ),
    };
  });
}
