import {
  Context,
  Effect,
  Exit,
  Layer,
  Option,
  PubSub,
  Ref,
  Scope,
  Stream,
} from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  type LocalProviderOffering,
  type ModelPackageEntry,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import {
  LocalModelAssessments,
  type LocalModelAssessmentsApi,
} from "./local-model-assessments"
import {
  LocalModelPackages,
  type LocalModelPackagesApi,
} from "./local-model-packages"
import {
  LocalProviderOfferingProjection,
  LocalProviderOfferingProjectionLive,
  type ProviderOfferingPackageEvidence,
  providerOfferingPackageEvidenceChanged,
  sameProviderOfferingPackageEvidence,
} from "./local-provider-offering-projection"
import {
  LocalProviderOfferings,
  type LocalProviderOfferingsApi,
} from "./local-provider-offerings"

const evidence = (
  installed: boolean,
  inspection: ProviderOfferingPackageEvidence[number]["packages"][number]["inspection"],
): ProviderOfferingPackageEvidence => [{
  providerModelId: ProviderModelIdSchema.make("test-configuration"),
  configurationId: ModelServingConfigurationIdSchema.make("configuration-test"),
  packages: [{
    packageId: ModelPackageIdSchema.make("package-test"),
    installed,
    inspection,
  }],
}]

describe("local provider offering package evidence", () => {
  it("compares equivalent availability evidence", () => {
    expect(sameProviderOfferingPackageEvidence(
      evidence(false, "Pending"),
      evidence(false, "Pending"),
    )).toBe(true)
  })

  it("changes when installation or inspection becomes authoritative", () => {
    expect(sameProviderOfferingPackageEvidence(
      evidence(false, "Pending"),
      evidence(true, "Pending"),
    )).toBe(false)
    expect(sameProviderOfferingPackageEvidence(
      evidence(true, "Pending"),
      evidence(true, "Inspected"),
    )).toBe(false)
  })

  it("does not treat equivalent configured-package evidence as a change", () => {
    expect(providerOfferingPackageEvidenceChanged(
      Option.some(evidence(false, "Pending")),
      evidence(false, "Pending"),
    )).toBe(false)
    expect(providerOfferingPackageEvidenceChanged(
      Option.some(evidence(false, "Pending")),
      evidence(true, "Pending"),
    )).toBe(true)
    expect(providerOfferingPackageEvidenceChanged(
      Option.none(),
      evidence(false, "Pending"),
    )).toBe(true)
  })

  it("ignores download progress but reassesses when package availability changes", async () => {
    const modelPackage = {
      id: ModelPackageIdSchema.make("package-progress"),
      files: [],
      source: { _tag: "HuggingFace", repository: "test/package-progress" },
      properties: { maximumContextLength: 131_072 },
    }
    const offering = {
      providerModelId: ProviderModelIdSchema.make("local-progress"),
      targetId: "target-progress",
      configuration: {
        id: ModelServingConfigurationIdSchema.make("configuration-progress"),
        target: { _tag: "Package", package: modelPackage },
        profile: { contextLength: 100_000 },
      },
      capabilities: {},
    } as unknown as LocalProviderOffering
    const downloadingEntry = (completedBytes: number) => ({
      package: modelPackage,
      targetId: Option.some(offering.targetId),
      localState: {
        _tag: "Downloading",
        attemptId: "attempt-progress",
        stage: "downloading",
        completedBytes,
        totalBytes: 100,
        bytesPerSecond: Option.none(),
      },
      inspection: { _tag: "Pending" },
    }) as unknown as ModelPackageEntry
    const installedEntry = {
      package: modelPackage,
      targetId: Option.some(offering.targetId),
      localState: { _tag: "Installed", path: "/models/progress.gguf" },
      inspection: { _tag: "Inspected", capabilities: {} },
    } as unknown as ModelPackageEntry

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const packageEntries = yield* Ref.make<readonly ModelPackageEntry[]>([
        downloadingEntry(10),
      ])
      const packageChanges = yield* PubSub.unbounded<{
        readonly revision: number
        readonly state: { readonly entries: readonly ModelPackageEntry[] }
      }>()
      const assessmentCalls = yield* Ref.make(0)
      const dependencies = Layer.mergeAll(
        Layer.succeed(IcnCatalog, IcnCatalog.of({
          get: Effect.succeed({ revision: 0, state: { models: [], diagnostics: [] } }),
          changes: Stream.never,
          ready: Effect.succeed(true),
          refresh: Effect.void,
        })),
        Layer.succeed(IcnHardware, IcnHardware.of({
          get: Effect.succeed({ revision: 0, state: { topology_fingerprint: "topology" } }),
          changes: Stream.never,
          initialized: Effect.succeed(true),
          refresh: Effect.void,
          assessmentChanges: Stream.never,
        } as never)),
        Layer.succeed(LocalModelAssessments, LocalModelAssessments.of({
          state: Effect.succeed(new Map()),
          changes: Stream.never,
          assess: () => Ref.update(assessmentCalls, (count) => count + 1).pipe(
            Effect.as([]),
          ),
        } satisfies LocalModelAssessmentsApi)),
        Layer.succeed(LocalModelPackages, LocalModelPackages.of({
          initialized: Effect.succeed(true),
          snapshot: Ref.get(packageEntries).pipe(
            Effect.map((entries) => ({ revision: 0, state: { entries } })),
          ),
          changes: Stream.fromPubSub(packageChanges),
          installedPackageIds: Effect.succeed(new Set()),
          admitTarget: () => Effect.die("unused"),
          cancelAttempts: () => Effect.die("unused"),
          dismissTargetFailure: () => Effect.die("unused"),
          removeTargetPackages: () => Effect.die("unused"),
        } satisfies LocalModelPackagesApi)),
        Layer.succeed(LocalProviderOfferings, LocalProviderOfferings.of({
          list: Effect.succeed([offering]),
          changes: Stream.never,
          resolve: () => Effect.die("unused"),
          save: () => Effect.die("unused"),
        } satisfies LocalProviderOfferingsApi)),
      )
      yield* Layer.build(
        LocalProviderOfferingProjectionLive.pipe(Layer.provide(dependencies)),
      )
      yield* Effect.sleep("50 millis")
      expect(yield* Ref.get(assessmentCalls)).toBe(1)

      yield* Ref.set(packageEntries, [downloadingEntry(50)])
      yield* PubSub.publish(packageChanges, {
        revision: 1,
        state: { entries: [downloadingEntry(50)] },
      })
      yield* Effect.sleep("50 millis")
      expect(yield* Ref.get(assessmentCalls)).toBe(1)

      yield* Ref.set(packageEntries, [installedEntry])
      yield* PubSub.publish(packageChanges, {
        revision: 2,
        state: { entries: [installedEntry] },
      })
      yield* Effect.sleep("50 millis")
      expect(yield* Ref.get(assessmentCalls)).toBe(2)
    })))
  })

  it("does not block service acquisition on model assessment", async () => {
    const modelPackage = {
      id: "package-installed",
      files: [],
      properties: { maximumContextLength: 131_072 },
    }
    const offering = {
      providerModelId: ProviderModelIdSchema.make("local-installed"),
      targetId: "target-installed",
      configuration: {
        id: ModelServingConfigurationIdSchema.make("configuration-installed"),
        target: { _tag: "Package", package: modelPackage },
        profile: { contextLength: 100_000 },
      },
      capabilities: {},
    } as unknown as LocalProviderOffering
    const entry = {
      package: modelPackage,
      targetId: Option.some(offering.targetId),
      localState: { _tag: "Installed", path: "/models/installed.gguf" },
      inspection: { _tag: "Inspected", capabilities: {} },
    } as unknown as ModelPackageEntry
    const dependencies = Layer.mergeAll(
      Layer.succeed(IcnCatalog, IcnCatalog.of({
        get: Effect.succeed({ revision: 0, state: { models: [], diagnostics: [] } }),
        changes: Stream.never,
        ready: Effect.succeed(true),
        refresh: Effect.void,
      })),
      Layer.succeed(IcnHardware, IcnHardware.of({
        get: Effect.succeed({ revision: 0, state: { topology_fingerprint: "topology" } }),
        changes: Stream.never,
        initialized: Effect.succeed(true),
        refresh: Effect.void,
        assessmentChanges: Stream.never,
      } as never)),
      Layer.succeed(LocalModelAssessments, LocalModelAssessments.of({
        state: Effect.succeed(new Map()),
        changes: Stream.never,
        assess: () => Effect.never,
      } satisfies LocalModelAssessmentsApi)),
      Layer.succeed(LocalModelPackages, LocalModelPackages.of({
        initialized: Effect.succeed(true),
        snapshot: Effect.succeed({ revision: 0, state: { entries: [entry] } }),
        changes: Stream.never,
        installedPackageIds: Effect.succeed(new Set(["package-installed"])),
        admitTarget: () => Effect.die("unused"),
        cancelAttempts: () => Effect.die("unused"),
        dismissTargetFailure: () => Effect.die("unused"),
        removeTargetPackages: () => Effect.die("unused"),
      } satisfies LocalModelPackagesApi)),
      Layer.succeed(LocalProviderOfferings, LocalProviderOfferings.of({
        list: Effect.succeed([offering]),
        changes: Stream.never,
        resolve: () => Effect.die("unused"),
        save: () => Effect.die("unused"),
      } satisfies LocalProviderOfferingsApi)),
    )

    const acquired = await Effect.runPromise(Effect.gen(function* () {
      const scope = yield* Scope.make()
      const context = yield* Layer.buildWithScope(
        LocalProviderOfferingProjectionLive.pipe(Layer.provide(dependencies)),
        scope,
      ).pipe(Effect.timeout("250 millis"))
      const service = Context.get(context, LocalProviderOfferingProjection)
      yield* Scope.close(scope, Exit.void)
      return service
    }))

    expect(acquired).toBeDefined()
  })
})
