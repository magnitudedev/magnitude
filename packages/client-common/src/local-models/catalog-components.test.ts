import { describe, expect, it } from "vitest"
import { makeSetupModel } from "../application/fixtures/model"
import { Option } from "effect"
import { catalogComponentDownloadLabel, catalogComponentName, catalogComponentOffers, catalogDrafterWithheldNote } from "./catalog-components"

const projector = { component: "Projector", downloadBytes: 1_034_945_421 } as const

describe("catalog component offers", () => {
  it("offers the projector a found exact target lacks as image input, downloaded alone", () => {
    const model = { ...makeSetupModel(true), missingComponents: [projector] }
    const offers = catalogComponentOffers(model)
    expect(offers).toEqual([projector])
    expect(offers.map((offer) => `${catalogComponentName(offer.component)} · ${catalogComponentDownloadLabel(offer)}`))
      .toEqual(["Image input · Download (1.03 GB)"])
  })

  it("offers nothing while a download of the model runs or before it is installed", () => {
    const installed = makeSetupModel(true)
    const updating = {
      ...installed,
      missingComponents: [projector],
      acquisitionState: {
        _tag: "Updating" as const,
        installation: installed.acquisitionState._tag === "Installed" ? installed.acquisitionState.installation : undefined as never,
        residencyState: { _tag: "Unloaded" as const },
        progress: { stage: "downloading" as const, completedBytes: 1, totalBytes: 2, bytesPerSecond: { _tag: "None" } as never },
      },
    }
    expect(catalogComponentOffers(updating as never)).toEqual([])
    expect(catalogComponentOffers({ ...makeSetupModel(false), missingComponents: [projector] })).toEqual([])
  })

  it("names a drafter whose bytes are already here without a size", () => {
    expect(catalogComponentName("Drafter")).toBe("Drafter")
    expect(catalogComponentDownloadLabel({ component: "Drafter", downloadBytes: 0 })).toBe("Download")
  })

  it("says when a catalog model runs without its drafter because it fits only without it", () => {
    expect(catalogDrafterWithheldNote({ ...makeSetupModel(true), drafterWithheld: true }))
      .toEqual(Option.some("Off - this model fits this computer only without it"))
    expect(catalogDrafterWithheldNote(makeSetupModel(true))).toEqual(Option.none())
  })
})
