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
  it("turns network access on with the key required in a container, taking only the key and host names from a saved block", () => {
    const block = { enabled: false, bind: "192.168.1.5", requireApiKey: false, allowedHosts: ["a.local"], apiKey: "mag-saved" }
    const saved = Option.some(block)
    const container = (key?: string, allowedHosts: ReadonlyArray<string> = []) => Option.some({ networkKey: Option.fromNullable(key), allowedHosts })
    const on = { enabled: true, requireApiKey: true }
    expect(containerNetworkConfig(Option.none(), Option.none())).toEqual(Option.none())
    expect(containerNetworkConfig(Option.none(), saved)).toEqual(saved)
    expect(containerNetworkConfig(container(), Option.none())).toEqual(Option.some({ ...on, allowedHosts: [] }))
    expect(containerNetworkConfig(container("from-the-environment"), Option.none()))
      .toEqual(Option.some({ ...on, allowedHosts: [], apiKey: "from-the-environment" }))
    expect(containerNetworkConfig(container(), saved)).toEqual(Option.some({ ...on, allowedHosts: ["a.local"], apiKey: "mag-saved" }))
    expect(containerNetworkConfig(container("from-the-environment"), saved))
      .toEqual(Option.some({ ...on, allowedHosts: ["a.local"], apiKey: "from-the-environment" }))
    expect(containerNetworkConfig(container(undefined, ["nas.local", "a.local"]), saved))
      .toEqual(Option.some({ ...on, allowedHosts: ["a.local", "nas.local"], apiKey: "mag-saved" }))
  })
})
