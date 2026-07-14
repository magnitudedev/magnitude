import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Effect } from "effect"
import type { MagnitudeStorageShape } from "@magnitudedev/storage"
import type { ProviderAuth } from "@magnitudedev/protocol"
import { resolveEndpointProviderAuthFromStorage, resolveProviderConfiguration } from "./shared-client"

function storageWithAuth(initial?: ProviderAuth) {
  const entries: Record<string, ProviderAuth> = initial ? { local: initial } : {}
  const storage = {
    auth: {
      get: (providerId: string) => Effect.succeed(entries[providerId]),
      loadAll: () => Effect.succeed({ ...entries }),
      set: (providerId: string, auth: ProviderAuth) => Effect.sync(() => {
        entries[providerId] = auth
      }),
    },
  } as unknown as MagnitudeStorageShape
  return { entries, storage }
}

const PROVIDER_ENV_KEYS = [
  "MAGNITUDE_API_KEY",
  "MAGNITUDE_LOCAL_API_KEY",
  "MAGNITUDE_USE_LOCAL",
  "OPENROUTER_API_KEY",
  "AI_GATEWAY_API_KEY",
  "DEEPSEEK_API_KEY",
  "ZAI_API_KEY",
  "ZHIPU_API_KEY",
  "MOONSHOT_API_KEY",
  "KIMI_API_KEY",
]

beforeEach(() => {
  for (const key of PROVIDER_ENV_KEYS) vi.stubEnv(key, "")
})

afterEach(() => {
  vi.unstubAllEnvs()
})

describe("endpoint provider auth resolution", () => {
  it("persists and returns a missing provider's default endpoint", async () => {
    const { entries, storage } = storageWithAuth()

    const resolved = await Effect.runPromise(
      resolveEndpointProviderAuthFromStorage(storage, "local", {
        endpoint: "http://127.0.0.1:8080",
      }),
    )

    expect(resolved).toEqual({ endpoint: "http://127.0.0.1:8080" })
    expect(entries.local).toEqual({
      type: "endpoint",
      endpoint: "http://127.0.0.1:8080",
    })
  })

  it("uses an explicit endpoint instead of the default", async () => {
    const { entries, storage } = storageWithAuth({
      type: "endpoint",
      endpoint: "http://127.0.0.1:9090",
    })

    const resolved = await Effect.runPromise(
      resolveEndpointProviderAuthFromStorage(storage, "local", {
        endpoint: "http://127.0.0.1:8080",
      }),
    )

    expect(resolved).toEqual({ endpoint: "http://127.0.0.1:9090" })
    expect(entries.local).toEqual({
      type: "endpoint",
      endpoint: "http://127.0.0.1:9090",
    })
  })
})

describe("provider configuration resolution", () => {
  it("seeds the Llama.cpp default once and preserves a user endpoint", async () => {
    const seeded = storageWithAuth()
    const first = await Effect.runPromise(resolveProviderConfiguration(seeded.storage))

    expect(seeded.entries.llamacpp).toEqual({
      type: "endpoint",
      endpoint: "http://127.0.0.1:8080",
    })
    expect(first.authSummaries.find((summary) => summary.providerId === "llamacpp")).toMatchObject({
      source: "default",
      endpoint: "http://127.0.0.1:8080",
    })

    const unchanged = await Effect.runPromise(resolveProviderConfiguration(seeded.storage))
    expect(unchanged.authSummaries.find(
      (summary) => summary.providerId === "llamacpp",
    )?.source).toBe("default")

    seeded.entries.llamacpp = { type: "endpoint", endpoint: "http://127.0.0.1:9090" }
    const edited = await Effect.runPromise(resolveProviderConfiguration(seeded.storage))

    expect(edited.connections.llamacpp?.endpoint).toBe("http://127.0.0.1:9090")
    expect(edited.authSummaries.find((summary) => summary.providerId === "llamacpp")?.source).toBe("file")
  })

  it("uses environment keys over auth.json without exposing the secret", async () => {
    const { storage } = storageWithAuth()
    const entries = (await Effect.runPromise(storage.auth.loadAll())) as Record<string, ProviderAuth>
    entries.openrouter = { type: "api", key: "file-secret" }
    const configured = storageWithAuth()
    configured.entries.openrouter = entries.openrouter
    vi.stubEnv("OPENROUTER_API_KEY", "env-secret")

    const resolved = await Effect.runPromise(resolveProviderConfiguration(configured.storage))
    const summary = resolved.authSummaries.find((candidate) => candidate.providerId === "openrouter")

    expect(resolved.connections.openrouter?.apiKey).toBe("env-secret")
    expect(summary).toMatchObject({ configured: true, source: "env" })
    expect(summary?.maskedKey).not.toContain("env-secret")
  })

  it("reports local Magnitude mode as environment-managed auth", async () => {
    const configured = storageWithAuth()
    vi.stubEnv("MAGNITUDE_USE_LOCAL", "true")
    vi.stubEnv("MAGNITUDE_LOCAL_API_KEY", "local-secret")

    const resolved = await Effect.runPromise(resolveProviderConfiguration(configured.storage))
    const summary = resolved.authSummaries.find((candidate) => candidate.providerId === "magnitude")

    expect(resolved.magnitudeApiKey).toBe("local-secret")
    expect(summary).toMatchObject({ configured: true, source: "env" })
  })

  it("does not accept legacy provider IDs or environment aliases", async () => {
    const configured = storageWithAuth()
    configured.entries.moonshotai = { type: "api", key: "legacy-kimi" }
    vi.stubEnv("ZHIPU_API_KEY", "legacy-zai")

    const resolved = await Effect.runPromise(resolveProviderConfiguration(configured.storage))

    expect(resolved.connections["kimi-api"]).toBeUndefined()
    expect(resolved.connections.zai).toBeUndefined()
  })
})
