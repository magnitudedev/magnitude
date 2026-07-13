import { describe, expect, it } from "vitest"
import { dot, lit, num, opt, sep, ver } from "../symbols"
import { type Family, match } from "../matcher"
import { atomizeModelId } from "../atomizer"

const llama3Family: Family = {
  familyId: "llama-3",
  patterns: [
    { pattern: [lit("llama"), sep(), opt("v"), lit("3"), dot(), ver()], priority: 100 },
    { pattern: [lit("llama"), sep(), opt("v"), lit("3")], priority: 90 },
  ],
}

const glm5Family: Family = {
  familyId: "glm-5",
  patterns: [
    { pattern: [lit("glm"), sep(), lit("5"), dot(), lit("1")], priority: 100 },
    { pattern: [lit("glm"), sep(), lit("5"), dot(), ver()], priority: 90 },
    { pattern: [lit("glm"), sep(), lit("5")], priority: 80 },
  ],
}

const qwen35Family: Family = {
  familyId: "qwen-3.5",
  patterns: [
    { pattern: [lit("qwen"), sep(), lit("3"), dot(), lit("5")], priority: 100 },
    { pattern: [lit("qwen"), sep(), lit("3"), dot(), lit("6")], priority: 100 },
  ],
}

const qwen3Family: Family = {
  familyId: "qwen-3",
  patterns: [
    { pattern: [lit("qwen"), sep(), lit("3")], priority: 80 },
  ],
}

const deepseekV3Family: Family = {
  familyId: "deepseek-v3",
  patterns: [
    { pattern: [lit("deepseek"), sep(), lit("r"), lit("1")], priority: 100, exclude: [lit("distill")] },
  ],
}

describe("matcher", () => {
  it("matches llama-3.3 variants", () => {
    const atoms = atomizeModelId("meta-llama/Llama-3.3-70B-Instruct")
    const result = match(atoms, [llama3Family])
    expect(result).toEqual({ familyId: "llama-3", priority: 100 })
  })

  it("matches Fireworks llama-v3p1", () => {
    const atoms = atomizeModelId("accounts/fireworks/models/llama-v3p1-8b-instruct")
    const result = match(atoms, [llama3Family])
    expect(result).toEqual({ familyId: "llama-3", priority: 100 })
  })

  it("matches Bedrock llama3-1-8b", () => {
    const atoms = atomizeModelId("meta.llama3-1-8b-instruct-v1:0")
    const result = match(atoms, [llama3Family])
    expect(result).toEqual({ familyId: "llama-3", priority: 90 })
  })

  it("does not match llama-4 with llama-3 pattern", () => {
    const atoms = atomizeModelId("llama-4-scout")
    const result = match(atoms, [llama3Family])
    expect(result).toBeNull()
  })

  it("avoids glm-5 false positive on glm-4p5", () => {
    const atoms = atomizeModelId("glm-4p5")
    const result = match(atoms, [glm5Family])
    expect(result).toBeNull()
  })

  it("matches glm-5.1", () => {
    const atoms = atomizeModelId("glm-5.1")
    const result = match(atoms, [glm5Family])
    expect(result).toEqual({ familyId: "glm-5", priority: 100 })
  })

  it("matches future glm-5.3 at priority 90", () => {
    const atoms = atomizeModelId("glm-5.3")
    const result = match(atoms, [glm5Family])
    expect(result).toEqual({ familyId: "glm-5", priority: 90 })
  })

  it("qwen-3.5 beats qwen-3 for qwen3.5", () => {
    const atoms = atomizeModelId("qwen3.5:35b")
    const result = match(atoms, [qwen35Family, qwen3Family])
    expect(result).toEqual({ familyId: "qwen-3.5", priority: 100 })
  })

  it("qwen3 without minor maps to qwen-3", () => {
    const atoms = atomizeModelId("qwen3:32b")
    const result = match(atoms, [qwen35Family, qwen3Family])
    expect(result).toEqual({ familyId: "qwen-3", priority: 80 })
  })

  it("exclusion prevents deepseek-r1-distill-llama matching deepseek-v3", () => {
    const atoms = atomizeModelId("deepseek-r1-distill-llama-70b")
    const result = match(atoms, [deepseekV3Family])
    expect(result).toBeNull()
  })

  it("deepseek-r1 without distill matches deepseek-v3", () => {
    const atoms = atomizeModelId("deepseek-r1")
    const result = match(atoms, [deepseekV3Family])
    expect(result).toEqual({ familyId: "deepseek-v3", priority: 100 })
  })
})
