import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { authorizesRemoteInference, generateApiKey, isAllowedHostHeader, isLoopbackAddress, LOOPBACK_ONLY, resolveNetworkAccess, type NetworkAccess } from "./network-access"

const enabled = (overrides: Partial<NetworkAccess> = {}): NetworkAccess => ({
  enabled: true, bind: "0.0.0.0", apiKey: Option.some("mag-secret"), requireApiKey: true, allowedHosts: [], warning: Option.none(), ...overrides,
})

describe("network access resolution", () => {
  it("is loopback only when absent or disabled", () => {
    expect(resolveNetworkAccess(Option.none())).toEqual(LOOPBACK_ONLY)
    expect(resolveNetworkAccess(Option.some({ enabled: false, bind: "0.0.0.0", apiKey: "k", requireApiKey: false, allowedHosts: ["a"] }))).toEqual(LOOPBACK_ONLY)
  })
  it("binds all interfaces unless a specific address is configured", () => {
    expect(resolveNetworkAccess(Option.some({ enabled: true, requireApiKey: true, allowedHosts: [] })).bind).toBe("0.0.0.0")
    expect(resolveNetworkAccess(Option.some({ enabled: true, bind: "100.64.1.5", requireApiKey: true, allowedHosts: [] })).bind).toBe("100.64.1.5")
    const bad = resolveNetworkAccess(Option.some({ enabled: true, bind: "my-mac", requireApiKey: true, allowedHosts: [] }))
    expect(bad.bind).toBe("0.0.0.0")
    expect(Option.isSome(bad.warning)).toBe(true)
  })
  it("generates distinct keys with a recognisable prefix", () => {
    const first = generateApiKey()
    expect(first.startsWith("mag-")).toBe(true)
    expect(first.length).toBeGreaterThan(20)
    expect(generateApiKey()).not.toBe(first)
  })
})

describe("loopback detection", () => {
  it("recognises IPv4, IPv6, and mapped loopback addresses", () => {
    for (const address of ["127.0.0.1", "127.0.0.53", "::1", "::ffff:127.0.0.1", "[::1]"]) expect(isLoopbackAddress(address)).toBe(true)
    for (const address of ["192.168.1.67", "100.124.15.114", "::ffff:192.168.1.2", "10.0.0.1"]) expect(isLoopbackAddress(address)).toBe(false)
  })
})

describe("Host header allowlist", () => {
  it("always accepts local names and nothing else while loopback only", () => {
    for (const host of ["localhost", "localhost:10100", "127.0.0.1:10100", "[::1]:10100"]) expect(isAllowedHostHeader(host, LOOPBACK_ONLY)).toBe(true)
    for (const host of ["192.168.1.67:10100", "evil.com", "host.docker.internal:10100", "mac.tail1234.ts.net", undefined]) expect(isAllowedHostHeader(host, LOOPBACK_ONLY)).toBe(false)
  })
  it("accepts IP literals, Docker, Tailscale, and configured names once enabled, never a wildcard", () => {
    const network = enabled({ allowedHosts: ["My-Mac.local"] })
    for (const host of ["192.168.1.67:10100", "100.124.15.114", "[fe80::1]:10100", "host.docker.internal:10100", "mac.tail1234.ts.net", "my-mac.local:10100"]) {
      expect(isAllowedHostHeader(host, network)).toBe(true)
    }
    for (const host of ["evil.com", "evil.com:10100", "ts.net", "attacker.ts.net.evil.com", "*"]) expect(isAllowedHostHeader(host, network)).toBe(false)
  })
})

describe("remote inference authorization", () => {
  it("accepts the key as a bearer token or x-api-key and rejects everything else", () => {
    const network = enabled()
    expect(authorizesRemoteInference({ authorization: "Bearer mag-secret" }, network)).toBe(true)
    expect(authorizesRemoteInference({ authorization: "bearer mag-secret" }, network)).toBe(true)
    expect(authorizesRemoteInference({ "x-api-key": "mag-secret" }, network)).toBe(true)
    expect(authorizesRemoteInference({ authorization: "Bearer magnitude-local" }, network)).toBe(false)
    expect(authorizesRemoteInference({ authorization: "Bearer mag-secre" }, network)).toBe(false)
    expect(authorizesRemoteInference({}, network)).toBe(false)
  })
  it("requires nothing when the key is switched off, and refuses when required but missing", () => {
    expect(authorizesRemoteInference({}, enabled({ requireApiKey: false }))).toBe(true)
    expect(authorizesRemoteInference({ authorization: "Bearer anything" }, enabled({ apiKey: Option.none() }))).toBe(false)
  })
})
