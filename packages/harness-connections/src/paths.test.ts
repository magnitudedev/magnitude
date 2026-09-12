import { expect, it } from "vitest"
import { harnessConnectionPaths } from "./paths"

  it("uses resolved configuration roots only outside isolated profiles", () => {
    const environment = { HERMES_HOME: "/resolved/hermes", CODEX_HOME: "/resolved/codex", PI_CODING_AGENT_DIR: "/resolved/pi" }
    expect(harnessConnectionPaths(undefined, environment).hermes).toBe("/resolved/hermes/config.yaml")
    expect(harnessConnectionPaths(undefined, environment).codexUser).toBe("/resolved/codex/config.toml")
    const isolated = harnessConnectionPaths("/isolated", environment)
    expect(isolated.hermes).toBe("/isolated/.hermes/config.yaml")
    expect(isolated.piSettings).toBe("/isolated/.pi/agent/settings.json")
  })
