import { describe, expect, it } from "vitest"
import type { RecommendableModel } from "@magnitudedev/acn-protocol"
import { installedCatalogConfigurations } from "./configuration-recovery"

const candidate = (
  id: string,
  packageId = "package-a",
): RecommendableModel => ({
  configuration: {
    id,
    bundle: { _tag: "Standalone", package: {
      id: packageId,
      source: { _tag: "Local", path: "/models" },
      files: [],
      relationships: [],
      properties: {
        format: "gguf",
        quantization: "Q4",
        quantizationName: "4-bit",
        architecture: "test",
        maximumContextLength: 65_536,
      },
    } },
    profile: { contextLength: 32_768 },
  },
} as unknown as RecommendableModel)

describe("configuration recovery candidates", () => {
  it("restores every catalog configuration whose exact packages are installed", () => {
    expect(installedCatalogConfigurations(
      [candidate("configuration-a")],
      new Set(["package-a"]),
    ).map(({ id }) => id)).toEqual(["configuration-a"])
    expect(installedCatalogConfigurations(
      [candidate("configuration-a"), candidate("configuration-b")],
      new Set(["package-a"]),
    ).map(({ id }) => id)).toEqual(["configuration-a", "configuration-b"])
    expect(installedCatalogConfigurations(
      [candidate("configuration-a")],
      new Set(),
    )).toEqual([])
  })
})
