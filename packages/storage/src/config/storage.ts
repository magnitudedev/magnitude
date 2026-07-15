import * as FileSystem from "@effect/platform/FileSystem";
import * as Path from "@effect/platform/Path";
import { type PlatformError } from "@effect/platform/Error";
import { Effect, Schema } from "effect";

import { makeStorageIo, type JsonError, SchemaDecodeError } from "../io/storage";
import { readStructuredFile } from "../io/structured-file";
import { GlobalStorage } from "../services";
import {
  MagnitudeConfigSchema,
  resolveContextLimitPolicy,
  type ContextLimitPolicy,
  type MagnitudeConfig,
} from "../types/config";
import type { ConfigStorageShape } from "./contracts";

const DEFAULT_CONFIG = Schema.decodeUnknownSync(MagnitudeConfigSchema)({});

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

    // Durable read: Missing is a legitimate first-run state (return defaults),
    // Invalid is corruption (return typed error — never silently overwrite).
    const readConfig = (): Effect.Effect<MagnitudeConfig, PlatformError | JsonError> =>
      Effect.gen(function* () {
        const result = yield* readStructuredFile(g.configFile, MagnitudeConfigSchema).pipe(
          Effect.provideService(FileSystem.FileSystem, fs)
        );
        if (result._tag === "Missing") return DEFAULT_CONFIG;
        if (result._tag === "Invalid") {
          return yield* Effect.fail(
            new SchemaDecodeError({ path: g.configFile, message: result.error.reason })
          );
        }
        return result.value;
      });

    return {
      load: () => readConfig(),

      save: (config) =>
        io.withPathLock(g.configFile, io.writeJsonFile(g.configFile, config)),

      update: (f) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfig();
            const next = f(current);
            yield* io.writeJsonFile(g.configFile, next);
            return next;
          })
        ),

      getContextLimitPolicy: () =>
        readConfig().pipe(Effect.map(resolveContextLimitPolicy)),

      setContextLimitPolicy: (policy) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfig();
            yield* io.writeJsonFile(g.configFile, {
              ...current,
              contextLimits: {
                ...(current.contextLimits ?? {}),
                ...policy,
              },
            });
          })
        ),

      getModelConfig: () =>
        readConfig().pipe(Effect.map((config) => config.models ?? null)),

      updateModelConfig: (slots) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfig();
            // Merge provided slots into existing; drop slots with no overrides.
            const existingSlots = current.models?.slots ?? {};
            const merged = { ...existingSlots };
            for (const [slotId, slotConfig] of Object.entries(slots)) {
              if (
                slotConfig.providerId ||
                slotConfig.providerModelId ||
                slotConfig.reasoningEffort
              ) {
                merged[slotId as keyof typeof merged] = slotConfig;
              } else {
                delete merged[slotId as keyof typeof merged];
              }
            }
            yield* io.writeJsonFile(g.configFile, {
              ...current,
              models: {
                ...current.models,
                slots: merged,
              },
            });
          })
        ),

      getOnboardingConfig: () =>
        readConfig().pipe(Effect.map((config) => config.onboarding ?? null)),

      getLocalInferenceConfig: () =>
        readConfig().pipe(Effect.map((config) => config.localInference ?? null)),

      setLocalInferenceConfig: (localInference) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfig();
            yield* io.writeJsonFile(g.configFile, {
              ...current,
              localInference,
            });
          })
        ),

      completeCliModelSetupOnboarding: (completedAt) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfig();
            yield* io.writeJsonFile(g.configFile, {
              ...current,
              onboarding: {
                ...current.onboarding,
                completedAt,
              },
            });
          })
        ),
    };
  });
}
