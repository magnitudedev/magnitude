import { describe, expect, it } from "vitest"
import { installationLoaderEnvironment, installationNativePath } from "./installation-environment.js"

describe("ICN installation loader environment", () => {
  it("clears ambient Unix library search paths", () => {
    expect(installationLoaderEnvironment("/runtime", "linux", "/ambient"))
      .toEqual({ LD_LIBRARY_PATH: "" })
    expect(installationLoaderEnvironment("/runtime", "darwin", "/ambient"))
      .toEqual({ DYLD_LIBRARY_PATH: "" })
  })

  it("prepends the owned runtime directory on Windows", () => {
    expect(installationLoaderEnvironment("C:\\runtime", "win32", "C:\\Windows"))
      .toEqual({ PATH: "\\\\?\\C:\\runtime;C:\\Windows" })
    expect(installationLoaderEnvironment("C:\\runtime", "win32", ""))
      .toEqual({ PATH: "\\\\?\\C:\\runtime" })
  })

  it("preserves extended local and network executable paths without changing Unix paths", () => {
    expect(installationNativePath("C:\\models\\inference.exe", "win32")).toBe("\\\\?\\C:\\models\\inference.exe")
    expect(installationNativePath("\\\\server\\models\\inference.exe", "win32")).toBe("\\\\?\\UNC\\server\\models\\inference.exe")
    expect(installationNativePath("\\\\?\\C:\\models\\inference.exe", "win32")).toBe("\\\\?\\C:\\models\\inference.exe")
    expect(installationNativePath("/models/inference", "linux")).toBe("/models/inference")
  })
})
