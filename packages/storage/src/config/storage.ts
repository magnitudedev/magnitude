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
  selectedSlotSelection,
  SlotSelectionStateSchema,
  unassignedSlotSelection,
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
    favoriteModels: config.models.favoriteModels,
    localProviderOfferings: config.models.localProviderOfferings,
    dismissedDownloadFailures: config.models.dismissedDownloadFailures,
  };
  const removedModelFields = config.models !== undefined
    && Reflect.ownKeys(config.models).some((key) =>
      key !== "slots"
      && key !== "localModelRecency"
      && key !== "favoriteModels"
      && key !== "localProviderOfferings"
      && key !== "dismissedDownloadFailures");
  const value = models === undefined ? { ...config } : { ...config, models };
  const removedLocalInference = Reflect.has(value, "localInference");
  if (removedLocalInference) Reflect.deleteProperty(value, "localInference");
  return { value, changed: removedModelFields || removedLocalInference };
};

const safeRecoveryMessage = (message: string): string =>
  message.replace(/, actual[\s\S]*$/, "").slice(0, 500);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const parseJsonOrUndefined = (text: string): unknown => {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
};

const normalizePersistedSlotStates = (value: unknown): boolean => {
  if (!isRecord(value) || !isRecord(value.models) || !isRecord(value.models.slots)) return false;
  let changed = false;
  for (const slotId of ["primary", "secondary"] as const) {
    const current = value.models.slots[slotId];
    const migrated = current === undefined
      ? unassignedSlotSelection()
      : isRecord(current) && current._tag !== "Unassigned" && current._tag !== "Selected"
        ? { _tag: "Selected", selection: current }
        : current;
    const normalized = Schema.is(SlotSelectionStateSchema)(migrated)
      ? migrated
      : unassignedSlotSelection();
    if (normalized !== current) {
      value.models.slots[slotId] = normalized;
      changed = true;
    }
  }
  return changed;
};

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
        const legacyText = yield* fs.readFileString(g.configFile).pipe(Effect.either);
        if (legacyText._tag === "Right") {
          const parsed = parseJsonOrUndefined(legacyText.right);
          if (parsed !== undefined && normalizePersistedSlotStates(parsed)) {
            yield* writeTextFileAtomic(g.configFile, `${JSON.stringify(parsed, null, 2)}\n`).pipe(
              Effect.provideService(FileSystem.FileSystem, fs),
            );
          }
        }
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

    const emptyModelConfig = () => ({
      slots: { primary: unassignedSlotSelection(), secondary: unassignedSlotSelection() },
      localModelRecency: { primary: [], secondary: [] },
      favoriteModels: [],
      localProviderOfferings: [],
      dismissedDownloadFailures: [],
    } as const);

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
            const existingModels = current.models ?? emptyModelConfig();
            yield* writeConfigUnlocked({
              ...current,
              models: {
                ...existingModels,
                slots: {
                  ...existingModels.slots,
                  [slotId]: Option.match(selection, {
                    onNone: unassignedSlotSelection,
                    onSome: selectedSlotSelection,
                  }),
                },
              },
            });
          })
        ),

      upsertLocalProviderOffering: (offering) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfigUnlocked();
            const existingModels = current.models ?? emptyModelConfig();
            const withoutExisting = existingModels.localProviderOfferings.filter(
              ({ providerModelId }) => providerModelId !== offering.providerModelId,
            );
            yield* writeConfigUnlocked({
              ...current,
              models: {
                ...existingModels,
                localProviderOfferings: [...withoutExisting, offering],
              },
            });
          }),
        ),

      dismissDownloadFailure: (packageId) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfigUnlocked();
            const existingModels = current.models ?? emptyModelConfig();
            yield* writeConfigUnlocked({
              ...current,
              models: {
                ...existingModels,
                dismissedDownloadFailures: [
                  ...new Set([...existingModels.dismissedDownloadFailures, packageId]),
                ],
              },
            });
          }),
        ),

      clearDismissedDownloadFailure: (packageId) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfigUnlocked();
            const existingModels = current.models ?? emptyModelConfig();
            yield* writeConfigUnlocked({
              ...current,
              models: {
                ...existingModels,
                dismissedDownloadFailures: existingModels.dismissedDownloadFailures.filter(
                  (candidate) => candidate !== packageId,
                ),
              },
            });
          }),
        ),

      getOnboardingConfig: () =>
        readConfig().pipe(Effect.map((config) => config.onboarding)),


      updateOnboardingState: (completed) =>
        io.withPathLock(
          g.configFile,
          Effect.gen(function* () {
            const current = yield* readConfigUnlocked();
            yield* writeConfigUnlocked({
              ...current,
              onboarding: Option.some({ completed }),
            });
          })
        ),
    };
  });
}
