import { describe, expect, it } from "vitest"
import { installationLoaderEnvironment } from "./installation-environment.js"

describe("ICN installation loader environment", () => {
  it("clears ambient Unix library search paths", () => {
    expect(installationLoaderEnvironment("/runtime", "linux", "/ambient"))
      .toEqual({ LD_LIBRARY_PATH: "" })
    expect(installationLoaderEnvironment("/runtime", "darwin", "/ambient"))
      .toEqual({ DYLD_LIBRARY_PATH: "" })
  })

  it("prepends the owned runtime directory on Windows", () => {
    expect(installationLoaderEnvironment("C:\\runtime", "win32", "C:\\Windows"))
      .toEqual({ PATH: "C:\\runtime;C:\\Windows" })
    expect(installationLoaderEnvironment("C:\\runtime", "win32", ""))
      .toEqual({ PATH: "C:\\runtime" })
  })
})
