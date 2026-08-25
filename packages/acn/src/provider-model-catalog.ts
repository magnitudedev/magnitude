import { FetchHttpClient } from "@effect/platform"
import { Cause, Context, Data, Effect, Either, Exit, Layer, Match, Option, Schema, Scope, Stream, SubscriptionRef } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  ProviderModelCatalogLoading,
  ProviderModelCatalogStateSchema,
  SECONDARY_SLOT_ID,
  type ProviderCatalogEntry,
  type ProviderCatalogFailure,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
} from "@magnitudedev/acn-protocol"
import {
  ProviderClient,
  type ProviderCatalogOutcome,
  type ProviderId,
  type ProviderModel,
  type ProviderRegistryInfo,
  type ReasoningEffort,
} from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { makeServiceOperationCoordinator } from "./service-operation-coordinator"

class ProviderContractViolation extends Data.TaggedError("ProviderContractViolation")<{
  readonly providerId: ProviderId
  readonly message: string
}> {}

type DiscoverableProperty<A> =
  | { readonly _tag: "Deferred" }
  | { readonly _tag: "Discovering" }
  | { readonly _tag: "Cached"; readonly value: A }
  | { readonly _tag: "Resolved"; readonly value: A }
  | { readonly _tag: "Refreshing"; readonly value: A }
  | { readonly _tag: "Failed" }

const resolvedProperty = <A>(property: DiscoverableProperty<A>): Option.Option<A> => {
  switch (property._tag) {
    case "Cached":
    case "Resolved":
    case "Refreshing":
      return Option.some(property.value)
    case "Deferred":
    case "Discovering":
    case "Failed":
      return Option.none()
  }
}

const toCatalogModel = (
  model: ProviderModel,
): Either.Either<ProviderModelCatalogEntry, ProviderContractViolation> => {
  const vision = resolvedProperty(model.properties.vision)
  if (Option.isNone(vision)) return Either.left(new ProviderContractViolation({
    providerId: model.providerId,
    message: `${model.displayName} has incomplete vision properties`,
  }))
  const reasoning = resolvedProperty(model.properties.reasoning)
  if (Option.isNone(reasoning)) return Either.left(new ProviderContractViolation({
    providerId: model.providerId,
    message: `${model.displayName} has incomplete reasoning properties`,
  }))
  const efforts: readonly ReasoningEffort[] = reasoning.value
  const supportedSlots = "slots" in model && Array.isArray(model.slots)
    ? model.slots.flatMap((slot) => slot === "primary"
      ? [PRIMARY_SLOT_ID]
      : slot === "secondary"
        ? [SECONDARY_SLOT_ID]
        : [])
    : [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID]
  return Either.right({
    providerId: model.providerId,
    providerModelId: model.providerModelId,
    modelFamilyId: Option.fromNullable(model.modelFamilyId),
    displayName: model.displayName,
    variantLabel: Option.none(),
    supportedSlots,
    contextWindow: model.contextWindow,
    maxOutputTokens: model.maxOutputTokens,
    memory: Option.none(),
    capabilities: {
      vision: vision.value,
      tools: model.servingCapabilities.tools,
      structuredOutput: model.servingCapabilities.structuredOutput,
      reasoning: {
        supported: efforts.length > 0,
        efforts,
        defaultEffort: efforts.length > 0 ? Option.some(model.defaultReasoningEffort) : Option.none(),
      },
    },
    availability: model.availability,
    pricing: Option.map(model.pricing, (pricing) => ({
      input: pricing.input,
      output: pricing.output,
      cachedInput: Option.fromNullable(pricing.cached_input),
    })),
  })
}

const providerEntry = (provider: ProviderRegistryInfo): ProviderCatalogEntry => ({
  providerId: provider.id,
  displayName: provider.displayName,
  kind: provider.kind,
  authentication: Match.value(provider.authStatus).pipe(
    Match.tag("authenticated", () => "Authenticated" as const),
    Match.tag("no_auth_required", () => "NotRequired" as const),
    Match.tag("not_configured", () => "NotConfigured" as const),
    Match.exhaustive,
  ),
  availability: Match.value(provider.status).pipe(
    Match.when("ok", () => ({ _tag: "Available" as const })),
    Match.when(undefined, () => ({ _tag: "Available" as const })),
    Match.when("loading", () => ({ _tag: "Loading" as const, message: Option.fromNullable(provider.message) })),
    Match.when("not_found", () => ({
      _tag: "NotFound" as const,
      message: Option.fromNullable(provider.message),
      hint: Option.fromNullable(provider.hint),
    })),
    Match.when("error", () => ({ _tag: "Failed" as const, message: provider.message ?? "Provider unavailable" })),
    Match.exhaustive,
  ),
})

const contents = (state: ProviderModelCatalogState) => ProviderModelCatalogLifecycle.match(state, {
  Loading: () => ({ providers: [] as readonly ProviderCatalogEntry[], models: [] as readonly ProviderModelCatalogEntry[], failures: [] as readonly ProviderCatalogFailure[] }),
  Ready: ({ providers, models }) => ({ providers, models, failures: [] as readonly ProviderCatalogFailure[] }),
  Refreshing: ({ providers, models, failures }) => ({ providers, models, failures }),
  Degraded: ({ providers, models, failures }) => ({ providers, models, failures }),
  Unavailable: ({ providers, failures }) => ({ providers, models: [] as readonly ProviderModelCatalogEntry[], failures }),
})

const providerFailure = (providerId: ProviderId, message: string): ProviderCatalogFailure => ({
  _tag: "ProviderFailure",
  providerId,
  message,
})

const catalogFailure = (message: string): ProviderCatalogFailure => ({ _tag: "CatalogFailure", message })

const outcomeModels = (
  outcome: ProviderCatalogOutcome,
): Either.Either<readonly ProviderModelCatalogEntry[], ProviderCatalogFailure> => {
  if (outcome._tag === "Failure") {
    return Either.left(providerFailure(outcome.providerId, outcome.failure.message))
  }
  const models: ProviderModelCatalogEntry[] = []
  for (const model of outcome.models) {
    const projected = toCatalogModel(model)
    if (Either.isLeft(projected)) return Either.left(providerFailure(
      projected.left.providerId,
      projected.left.message,
    ))
    models.push(projected.right)
  }
  return Either.right(models)
}

export interface ProviderModelCatalogApi {
  readonly state: Effect.Effect<ProviderModelCatalogState>
  readonly changes: Stream.Stream<ProviderModelCatalogState>
  readonly refresh: (providerId: Option.Option<ProviderId>) => Effect.Effect<void>
}

export class ProviderModelCatalog extends Context.Tag("ProviderModelCatalog")<
  ProviderModelCatalog,
  ProviderModelCatalogApi
>() {}

const sameRefreshTarget = (
  left: Option.Option<ProviderId>,
  right: Option.Option<ProviderId>,
): boolean =>
  Option.isNone(left)
    ? Option.isNone(right)
    : Option.isSome(right) && left.value === right.value

export const ProviderModelCatalogLive: Layer.Layer<
  ProviderModelCatalog,
  never,
  ProviderClient | LocalProviderOfferings
> = Layer.scoped(ProviderModelCatalog, Effect.gen(function* () {
  const client = yield* ProviderClient
  const localOfferings = yield* LocalProviderOfferings
  const scope = yield* Scope.Scope
  const lock = yield* Effect.makeSemaphore(1)
  const stateLock = yield* Effect.makeSemaphore(1)
  const refreshOperations = yield* makeServiceOperationCoordinator<
    Option.Option<ProviderId>,
    never
  >(sameRefreshTarget)
  const current = yield* SubscriptionRef.make<ProviderModelCatalogState>(
    new ProviderModelCatalogLoading({}),
  )
  const equivalent = Schema.equivalence(ProviderModelCatalogStateSchema)

  const updateCatalog = (update: (state: ProviderModelCatalogState) => ProviderModelCatalogState) =>
    stateLock.withPermits(1)(Effect.gen(function* () {
      const state = yield* SubscriptionRef.get(current)
      const next = update(state)
      if (!equivalent(state, next)) yield* SubscriptionRef.set(current, next)
    }))

  const beginRefresh = updateCatalog((state) => ProviderModelCatalogLifecycle.match(state, {
    Loading: (current) => current,
    Ready: (current) => ProviderModelCatalogLifecycle.transition(current, "Refreshing", { failures: [] }),
    Refreshing: (current) => current,
    Degraded: (current) => ProviderModelCatalogLifecycle.transition(current, "Refreshing", {}),
    Unavailable: (current) => ProviderModelCatalogLifecycle.transition(current, "Refreshing", { models: [] }),
  }))

  const publish = (
    providers: readonly ProviderCatalogEntry[],
    models: readonly ProviderModelCatalogEntry[],
    failures: readonly ProviderCatalogFailure[],
  ) => updateCatalog((state) => {
    if (state._tag !== "Loading" && state._tag !== "Refreshing") return state
    if (failures.length === 0) return ProviderModelCatalogLifecycle.transition(state, "Ready", { providers, models })
    return ProviderModelCatalogLifecycle.transition(state, "Degraded", { providers, models, failures })
  })

  const reconcile = (outcomes: readonly ProviderCatalogOutcome[]) => Effect.gen(function* () {
    const previous = contents(yield* SubscriptionRef.get(current))
    const modelsByProvider = new Map<ProviderId, readonly ProviderModelCatalogEntry[]>()
    for (const model of previous.models) {
      modelsByProvider.set(model.providerId, [...(modelsByProvider.get(model.providerId) ?? []), model])
    }
    const failuresByProvider = new Map<ProviderId, ProviderCatalogFailure>()
    for (const failure of previous.failures) {
      if (failure._tag === "ProviderFailure") failuresByProvider.set(failure.providerId, failure)
    }
    const generalFailures: ProviderCatalogFailure[] = []

    for (const outcome of outcomes) {
      if (outcome.providerId === LOCAL_PROVIDER_ID) continue
      const projected = outcomeModels(outcome)
      if (Either.isLeft(projected)) failuresByProvider.set(outcome.providerId, projected.left)
      else {
        modelsByProvider.set(outcome.providerId, projected.right)
        failuresByProvider.delete(outcome.providerId)
      }
    }

    const localOfferingsResult = yield* Effect.either(localOfferings.catalog)
    const localModels = Either.isRight(localOfferingsResult) ? localOfferingsResult.right : []
    modelsByProvider.set(LOCAL_PROVIDER_ID, localModels)
    if (Either.isLeft(localOfferingsResult)) {
      failuresByProvider.set(LOCAL_PROVIDER_ID, providerFailure(
        LOCAL_PROVIDER_ID,
        localOfferingsResult.left.message,
      ))
    } else {
      failuresByProvider.delete(LOCAL_PROVIDER_ID)
    }

    const providerResult = yield* client.listProviders.pipe(
      Effect.provide(FetchHttpClient.layer),
      Effect.either,
    )
    const providers = Either.match(providerResult, {
      onLeft: (error) => {
        generalFailures.push(catalogFailure(String(error)))
        return previous.providers
      },
      onRight: (values) => values.map(providerEntry),
    })
    const providersById = new Map(providers.map((provider) => [provider.providerId, provider]))
    const projectedModels = [...modelsByProvider.values()].flat().flatMap((model) => {
      const provider = providersById.get(model.providerId)
      if (!provider) return []
      const providerUnavailable = failuresByProvider.has(model.providerId)
        || provider.authentication === "NotConfigured"
        || provider.availability._tag !== "Available"
      return [{
        ...model,
        availability: providerUnavailable
          ? { _tag: "Disabled" as const, reason: "provider_unavailable" as const }
          : model.availability,
      }]
    })
    yield* publish(
      providers,
      projectedModels,
      [...failuresByProvider.values(), ...generalFailures],
    )
  })

  const refreshNow = (force: boolean, providerId: Option.Option<ProviderId>) =>
    lock.withPermits(1)(Effect.gen(function* () {
      yield* beginRefresh
      const result = yield* (force
        ? client.catalogs.refresh(Option.getOrUndefined(providerId))
        : client.catalogs.list).pipe(
          Effect.provide(FetchHttpClient.layer),
          Effect.either,
        )
      if (Either.isRight(result)) {
        yield* reconcile(result.right)
        return
      }
      const previous = contents(yield* SubscriptionRef.get(current))
      yield* publish(
        previous.providers,
        previous.models,
        [
          ...previous.failures.filter((failure) => failure._tag === "ProviderFailure"),
          catalogFailure(String(result.left)),
        ],
      )
    }))

  yield* refreshNow(false, Option.none())

  const terminalizeRefreshFailure = (cause: Cause.Cause<unknown>) => Effect.gen(function* () {
    yield* beginRefresh
    const previous = contents(yield* SubscriptionRef.get(current))
    yield* publish(
      previous.providers,
      previous.models,
      [
        ...previous.failures.filter((failure) => failure._tag === "ProviderFailure"),
        catalogFailure(Cause.pretty(cause).slice(0, 1_000)),
      ],
    )
  })

  const admitRefresh = (
    providerId: Option.Option<ProviderId>,
  ) => refreshOperations.admit(Effect.succeed({
    key: providerId,
    whenIdle: Effect.succeed(Option.some({
      commit: beginRefresh,
      operation: refreshNow(true, providerId),
      terminalize: (exit: Exit.Exit<void, never>) =>
        Effect.gen(function* () {
          if (Exit.isFailure(exit)) {
            yield* lock.withPermits(1)(terminalizeRefreshFailure(exit.cause))
          }
        }),
    })),
  })).pipe(Effect.orDie)

  const requestRefresh = (
    providerId: Option.Option<ProviderId>,
  ): Effect.Effect<void> => Effect.suspend(() => admitRefresh(providerId)).pipe(
    Effect.flatMap((admission) =>
      admission._tag === "Satisfied" ? Effect.void : admission.outcome.pipe(
        Effect.zipRight(admission._tag === "Conflicting"
          ? Effect.suspend(() => requestRefresh(providerId))
          : Effect.void),
      )),
  )

  const reconcileLocalProjection = lock.withPermits(1)(Effect.gen(function* () {
    yield* beginRefresh
    yield* reconcile([])
  })).pipe(
    Effect.catchAllCause((cause) =>
      lock.withPermits(1)(terminalizeRefreshFailure(cause)).pipe(
        Effect.zipRight(Effect.logError("Unable to reconcile local provider offerings").pipe(
          Effect.annotateLogs({ cause: Cause.pretty(cause).slice(0, 1_000) }),
        )),
      )),
  )

  yield* Effect.forkIn(
    localOfferings.catalogChanges.pipe(
      // A local offering change only reprojects cached catalog outcomes. It
      // must not force remote catalog refresh or hold background activity.
      Stream.runForEach(() => reconcileLocalProjection),
    ),
    scope,
  )

  return ProviderModelCatalog.of({
    state: SubscriptionRef.get(current),
    changes: current.changes,
    refresh: requestRefresh,
  })
}))
