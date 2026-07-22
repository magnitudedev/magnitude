import { FetchHttpClient } from "@effect/platform"
import { Context, Data, Effect, Either, Layer, Match, Option, Schema, Scope, Stream } from "effect"
import {
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  ProviderModelCatalogLoading,
  ProviderModelCatalogMirror,
  SECONDARY_SLOT_ID,
  type MirroredSnapshot,
  type ProviderCatalogEntry,
  type ProviderCatalogFailure,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
} from "@magnitudedev/protocol"
import {
  ProviderClient,
  type ProviderCatalogOutcome,
  type ProviderId,
  type ProviderModel,
  type ProviderRegistryInfo,
  type ReasoningEffort,
} from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { makeMirroredState, MirroredStateChanges } from "./mirrored-state"
import { LocalModelInventory } from "./local-model-inventory"

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
    supportedSlots,
    contextWindow: model.contextWindow,
    maxOutputTokens: model.maxOutputTokens,
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
    pricing: Option.map(Option.fromNullable(model.pricing), (pricing) => ({
      input: pricing.input,
      output: pricing.output,
      cachedInput: Option.fromNullable(pricing.cached_input),
    })),
  })
}

const providerEntry = (provider: ProviderRegistryInfo): ProviderCatalogEntry => ({
  providerId: provider.id,
  displayName: provider.displayName,
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
  readonly snapshot: Effect.Effect<MirroredSnapshot<ProviderModelCatalogState>>
  readonly changes: Stream.Stream<MirroredSnapshot<ProviderModelCatalogState>>
  readonly refresh: (providerId: Option.Option<ProviderId>) => Effect.Effect<void>
}

export class ProviderModelCatalog extends Context.Tag("ProviderModelCatalog")<
  ProviderModelCatalog,
  ProviderModelCatalogApi
>() {}

export const ProviderModelCatalogLive: Layer.Layer<
  ProviderModelCatalog,
  never,
  ProviderClient | LocalModelInventory | MirroredStateChanges
> = Layer.scoped(ProviderModelCatalog, Effect.gen(function* () {
  const client = yield* ProviderClient
  const localInventory = yield* LocalModelInventory
  const scope = yield* Scope.Scope
  const lock = yield* Effect.makeSemaphore(1)
  const mirror = yield* makeMirroredState(ProviderModelCatalogMirror, new ProviderModelCatalogLoading({}))
  const equivalent = Schema.equivalence(ProviderModelCatalogMirror.stateSchema)

  const updateCatalog = (update: (state: ProviderModelCatalogState) => ProviderModelCatalogState) =>
    mirror.modify((state) => {
      const next = update(state)
      return { state: next, result: undefined, changed: !equivalent(state, next) }
    })

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
    if (models.length > 0) return ProviderModelCatalogLifecycle.transition(state, "Degraded", { providers, models, failures })
    return ProviderModelCatalogLifecycle.transition(state, "Unavailable", { providers, failures })
  })

  const reconcile = (outcomes: readonly ProviderCatalogOutcome[]) => Effect.gen(function* () {
    const previous = contents((yield* mirror.get).state)
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

    const localState = (yield* localInventory.snapshot).state
    if (localState._tag === "Ready") {
      modelsByProvider.set(LOCAL_PROVIDER_ID, yield* localInventory.localCatalog)
      failuresByProvider.delete(LOCAL_PROVIDER_ID)
    } else {
      failuresByProvider.set(LOCAL_PROVIDER_ID, providerFailure(
        LOCAL_PROVIDER_ID,
        localState._tag === "Failed" ? localState.error.message : "Local model inventory is loading",
      ))
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
      const previous = contents((yield* mirror.get).state)
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
  yield* Effect.forkIn(localInventory.changes.pipe(
    Stream.drop(1),
    Stream.runForEach(() => lock.withPermits(1)(Effect.gen(function* () {
      yield* beginRefresh
      yield* reconcile([])
    }))),
  ), scope)

  return ProviderModelCatalog.of({
    snapshot: mirror.get,
    changes: mirror.changes,
    refresh: (providerId) => refreshNow(true, providerId),
  })
}))
