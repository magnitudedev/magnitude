import { Context, Effect, Exit, Layer, Option, Scope, Stream } from "effect"
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
