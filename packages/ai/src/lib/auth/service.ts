import { Context, Effect, Layer, Option } from "effect"
import { NotConfigured } from "../errors/model-error"
import type { ProviderDefinition } from "../execution/provider-definition"
import { resolveEnvAuth } from "./env"
import { AuthStorage, type StoredAuth } from "./storage"
import type { ResolvedAuth } from "./types"

function storedToResolvedAuth(stored: StoredAuth): ResolvedAuth {
  switch (stored._tag) {
    case "api-key":
      return {
        _tag: "ApiKeyAuth",
        apiKey: stored.key,
      }
    case "oauth":
      return {
        _tag: "OAuthAuth",
        accessToken: stored.accessToken,
      }
  }
}

function resolveNoAuth(provider: ProviderDefinition): ResolvedAuth | null {
  return provider.authMethods.some((method) => method.type === "none")
    ? { _tag: "NoAuth" }
    : null
}

export class ModelAuth extends Context.Tag("@magnitudedev/ai/ModelAuth")<
  ModelAuth,
  {
    readonly resolveAuth: (provider: ProviderDefinition) => Effect.Effect<ResolvedAuth | null>
    readonly requireAuth: (provider: ProviderDefinition) => Effect.Effect<ResolvedAuth, NotConfigured>
  }
>() {}

export const ModelAuthLive = Layer.effect(
  ModelAuth,
  Effect.gen(function* () {
    const authStorage = yield* Effect.serviceOption(AuthStorage)

    const resolveStoredAuth = (providerId: string): Effect.Effect<ResolvedAuth | null> =>
      Option.match(authStorage, {
        onNone: () => Effect.succeed<ResolvedAuth | null>(null),
        onSome: (storage) =>
          storage.getAuth(providerId).pipe(
            Effect.map((stored) => (stored ? storedToResolvedAuth(stored) : null)),
          ),
      })

    const resolveAuth = (provider: ProviderDefinition): Effect.Effect<ResolvedAuth | null> =>
      Effect.gen(function* () {
        const envAuth = resolveEnvAuth(provider)
        if (envAuth) return envAuth

        const storedAuth = yield* resolveStoredAuth(provider.id)
        if (storedAuth) return storedAuth

        return resolveNoAuth(provider)
      })

    const requireAuth = (provider: ProviderDefinition): Effect.Effect<ResolvedAuth, NotConfigured> =>
      resolveAuth(provider).pipe(
        Effect.flatMap((auth) =>
          auth
            ? Effect.succeed(auth)
            : Effect.fail(
                new NotConfigured({
                  providerId: provider.id,
                  message: `No auth configured for provider "${provider.id}"`,
                }),
              ),
        ),
      )

    return {
      resolveAuth,
      requireAuth,
    }
  }),
)
