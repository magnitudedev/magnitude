import { describe, expect, it } from "vitest"

import { projectPackageAcquisition } from "./local-model-packages"
describe("projectPackageAcquisition", () => {
  it("preserves generated installation provenance", () => {
    expect(projectPackageAcquisition({
      _tag: "Installed",
      path: "/hf/model.gguf",
      origin: "HuggingFaceCache",
    })).toEqual({
      _tag: "Installed",
      path: "/hf/model.gguf",
      origin: "HuggingFaceCache",
    })
  })

})
