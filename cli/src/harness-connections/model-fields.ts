import type { HarnessModel } from "./contract"

export const modelInput = (model: HarnessModel): ReadonlyArray<"text" | "image"> =>
  model.capabilities.vision ? ["text", "image"] : ["text"]

export const modelMaxTokens = (model: HarnessModel): number => model.maxOutputTokens

export const zeroCost = () => ({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 })
