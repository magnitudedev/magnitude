import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { containerNetworkConfig, readContainerInstallation } from "./container"

describe("container installation", () => {
  it("is read from the image's marker, with a usable MAGNITUDE_NETWORK_KEY", async () => {
    const resources = await mkdtemp(join(tmpdir(), "magnitude-container-"))
    const read = (environment: Record<string, string>) => Effect.runPromise(readContainerInstallation(environment, resources).pipe(Effect.provide(BunContext.layer)))
    try {
      expect(await read({ MAGNITUDE_NETWORK_KEY: "a-key-from-the-environment" })).toEqual(Option.none())
      await writeFile(join(resources, "update-package.json"), '{"format":"deb"}')
      expect(await read({})).toEqual(Option.none())
      await writeFile(join(resources, "update-package.json"), '{"format":"docker"}')
      expect(await read({})).toEqual(Option.some({ networkKey: Option.none(), allowedHosts: [] }))
      expect(await read({ MAGNITUDE_NETWORK_KEY: "  a-key-from-the-environment  " })).toEqual(Option.some({ networkKey: Option.some("a-key-from-the-environment"), allowedHosts: [] }))
      expect(await read({ MAGNITUDE_ALLOWED_HOSTS: " NAS.local, ,magnitude.lan " })).toEqual(Option.some({ networkKey: Option.none(), allowedHosts: ["nas.local", "magnitude.lan"] }))
      for (const rejected of ["short", "has spaces in the middle", "   "]) {
        expect(await read({ MAGNITUDE_NETWORK_KEY: rejected })).toEqual(Option.some({ networkKey: Option.none(), allowedHosts: [] }))
      }
    } finally { await rm(resources, { recursive: true, force: true }) }
  })
  it("defaults network access on only in a container, where a saved block still decides and the environment's key wins", () => {
    const block = { enabled: false, requireApiKey: false, allowedHosts: ["a.local"], apiKey: "mag-saved" }
    const saved = Option.some(block)
    expect(containerNetworkConfig(Option.none(), Option.none())).toEqual(Option.none())
    expect(containerNetworkConfig(Option.none(), saved)).toEqual(saved)
    expect(containerNetworkConfig(Option.some({ networkKey: Option.none(), allowedHosts: [] }), Option.none()))
      .toEqual(Option.some({ enabled: true, requireApiKey: true, allowedHosts: [] }))
    expect(containerNetworkConfig(Option.some({ networkKey: Option.some("from-the-environment"), allowedHosts: [] }), Option.none()))
      .toEqual(Option.some({ enabled: true, requireApiKey: true, allowedHosts: [], apiKey: "from-the-environment" }))
    expect(containerNetworkConfig(Option.some({ networkKey: Option.some("from-the-environment"), allowedHosts: [] }), saved))
      .toEqual(Option.some({ ...block, apiKey: "from-the-environment" }))
    expect(containerNetworkConfig(Option.some({ networkKey: Option.none(), allowedHosts: ["nas.local", "a.local"] }), saved))
      .toEqual(Option.some({ ...block, allowedHosts: ["a.local", "nas.local"] }))
  })
})
