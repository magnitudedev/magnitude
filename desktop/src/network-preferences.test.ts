import { NodeContext } from "@effect/platform-node"
import { Effect, Option } from "effect"
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { expect, it } from "vitest"
import { listNetworkInterfaces, makeNetworkPreferences, networkAccessEquals, LOOPBACK_ONLY } from "@magnitudedev/daemon-management/desktop-native"

it("is loopback only until enabled, then generates a key once and keeps other configuration", async () => {
  const directory = await mkdtemp(join(tmpdir(), "magnitude-network-"))
  const path = join(directory, "config.json")
  try {
    const preferences = await Effect.runPromise(makeNetworkPreferences(directory).pipe(Effect.provide(NodeContext.layer)))
    const initial = await Effect.runPromise(preferences.read)
    expect(initial.saved).toEqual(Option.none())
    expect(initial.resolved).toEqual(LOOPBACK_ONLY)
    await writeFile(path, JSON.stringify({ appearance: "dark", extra: { value: 42 } }))
    await Effect.runPromise(preferences.update({ enabled: true }))
    const enabled = await Effect.runPromise(preferences.read)
    expect(enabled.resolved.enabled).toBe(true)
    expect(enabled.resolved.bind).toBe("0.0.0.0")
    expect(enabled.resolved.requireApiKey).toBe(true)
    const key = Option.getOrThrow(enabled.resolved.apiKey)
    expect(key.startsWith("mag-")).toBe(true)
    expect(JSON.parse(await readFile(path, "utf8"))).toMatchObject({ appearance: "dark", extra: { value: 42 }, network: { enabled: true, apiKey: key } })
    await Effect.runPromise(preferences.update({ bind: "192.168.1.10", requireApiKey: false }))
    const bound = await Effect.runPromise(preferences.read)
    expect(bound.resolved.bind).toBe("192.168.1.10")
    expect(bound.resolved.requireApiKey).toBe(false)
    expect(Option.getOrThrow(bound.resolved.apiKey)).toBe(key)
    await Effect.runPromise(preferences.regenerateApiKey)
    const regenerated = await Effect.runPromise(preferences.read)
    expect(Option.getOrThrow(regenerated.resolved.apiKey)).not.toBe(key)
    await Effect.runPromise(preferences.update({ bind: null, enabled: false }))
    const disabled = await Effect.runPromise(preferences.read)
    expect(disabled.resolved).toEqual(LOOPBACK_ONLY)
    expect(Option.getOrThrow(disabled.saved).apiKey).toBe(Option.getOrThrow(regenerated.resolved.apiKey))
    expect((await Effect.runPromise(preferences.update({ bind: "my-mac" }).pipe(Effect.either)))._tag).toBe("Left")
  } finally { await rm(directory, { recursive: true, force: true }) }
})

it("compares what the service enforces", () => {
  expect(networkAccessEquals(LOOPBACK_ONLY, { ...LOOPBACK_ONLY, allowedHosts: [] })).toBe(true)
  expect(networkAccessEquals(LOOPBACK_ONLY, { ...LOOPBACK_ONLY, enabled: true, bind: "0.0.0.0" })).toBe(false)
  expect(networkAccessEquals({ ...LOOPBACK_ONLY, apiKey: Option.some("a") }, { ...LOOPBACK_ONLY, apiKey: Option.some("b") })).toBe(false)
})

it("lists external IPv4 addresses and labels Tailscale's range", () => {
  const listed = listNetworkInterfaces({
    lo0: [{ address: "127.0.0.1", netmask: "255.0.0.0", family: "IPv4", mac: "00:00:00:00:00:00", internal: true, cidr: "127.0.0.1/8" }],
    en0: [
      { address: "192.168.1.67", netmask: "255.255.255.0", family: "IPv4", mac: "00:00:00:00:00:01", internal: false, cidr: "192.168.1.67/24" },
      { address: "fe80::1", netmask: "ffff:ffff:ffff:ffff::", family: "IPv6", mac: "00:00:00:00:00:01", internal: false, cidr: "fe80::1/64", scopeid: 1 },
    ],
    utun4: [{ address: "100.124.15.114", netmask: "255.255.255.255", family: "IPv4", mac: "00:00:00:00:00:02", internal: false, cidr: "100.124.15.114/32" }],
    awdl0: [{ address: "169.254.10.2", netmask: "255.255.0.0", family: "IPv4", mac: "00:00:00:00:00:03", internal: false, cidr: "169.254.10.2/16" }],
  })
  expect(listed).toEqual([
    { name: "en0", address: "192.168.1.67", tailscale: false },
    { name: "utun4", address: "100.124.15.114", tailscale: true },
  ])
})
