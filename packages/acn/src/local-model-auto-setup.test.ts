import { Context, Effect, Exit, Layer, Option, Scope, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelOfferingTargetIdSchema,
  type ModelPackageEntry,
} from "@magnitudedev/acn-protocol"
import { IcnCatalog, IcnHardware } from "@magnitudedev/icn"
import {
  LocalModelAssessments,
  type LocalModelAssessmentsApi,
} from "./local-model-assessments"
import {
  LocalModelAutoSetup,
  LocalModelAutoSetupLive,
} from "./local-model-auto-setup"
import {
  LocalModelPackages,
  type LocalModelPackagesApi,
} from "./local-model-packages"
import {
  LocalProviderOfferings,
  type LocalProviderOfferingsApi,
} from "./local-provider-offerings"

describe("LocalModelAutoSetupLive", () => {
  it("does not block service acquisition on installed-model assessment", async () => {
    const targetId = ModelOfferingTargetIdSchema.make("target-installed")
    const entry = {
      package: {
        id: "package-installed",
        files: [],
        properties: { maximumContextLength: 131_072 },
      },
      targetId: Option.some(targetId),
      localState: { _tag: "Installed", path: "/models/installed.gguf" },
      inspection: { _tag: "Inspected", capabilities: {} },
    } as unknown as ModelPackageEntry
    const assessments = LocalModelAssessments.of({
      state: Effect.succeed(new Map()),
      changes: Stream.never,
      assess: () => Effect.never,
    } satisfies LocalModelAssessmentsApi)
    const packages = LocalModelPackages.of({
      snapshot: Effect.succeed({ revision: 0, state: { entries: [entry] } }),
      changes: Stream.never,
      installedPackageIds: Effect.succeed(new Set(["package-installed"])),
      admitTarget: () => Effect.die("unused"),
      cancelAttempts: () => Effect.die("unused"),
      dismissTargetFailure: () => Effect.die("unused"),
      removeTargetPackages: () => Effect.die("unused"),
    } satisfies LocalModelPackagesApi)
    const offerings = LocalProviderOfferings.of({
      list: Effect.succeed([]),
      changes: Stream.never,
      resolve: () => Effect.die("unused"),
      save: () => Effect.die("unused"),
    } satisfies LocalProviderOfferingsApi)
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
      Layer.succeed(LocalModelAssessments, assessments),
      Layer.succeed(LocalModelPackages, packages),
      Layer.succeed(LocalProviderOfferings, offerings),
    )

    const acquired = await Effect.runPromise(Effect.gen(function* () {
      const scope = yield* Scope.make()
      const context = yield* Layer.buildWithScope(
        LocalModelAutoSetupLive.pipe(Layer.provide(dependencies)),
        scope,
      ).pipe(Effect.timeout("250 millis"))
      const service = Context.get(context, LocalModelAutoSetup)
      yield* Scope.close(scope, Exit.void)
      return service
    }))

    expect(acquired).toBeDefined()
  })
})
