import { describe, expect, it } from "vitest"
import {
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
} from "@magnitudedev/protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import {
  type ProviderOfferingPackageEvidence,
  sameProviderOfferingPackageEvidence,
} from "./local-provider-offering-projection"

const evidence = (
  installed: boolean,
  inspection: ProviderOfferingPackageEvidence[number]["packages"][number]["inspection"],
): ProviderOfferingPackageEvidence => [{
  providerModelId: ProviderModelIdSchema.make("test-configuration"),
  configurationId: ModelServingConfigurationIdSchema.make("configuration-test"),
  packages: [{
    packageId: ModelPackageIdSchema.make("package-test"),
    installed,
    inspection,
  }],
}]

describe("local provider offering package evidence", () => {
  it("compares equivalent availability evidence", () => {
    expect(sameProviderOfferingPackageEvidence(
      evidence(false, "Pending"),
      evidence(false, "Pending"),
    )).toBe(true)
  })

  it("changes when installation or inspection becomes authoritative", () => {
    expect(sameProviderOfferingPackageEvidence(
      evidence(false, "Pending"),
      evidence(true, "Pending"),
    )).toBe(false)
    expect(sameProviderOfferingPackageEvidence(
      evidence(true, "Pending"),
      evidence(true, "Inspected"),
    )).toBe(false)
  })
})
