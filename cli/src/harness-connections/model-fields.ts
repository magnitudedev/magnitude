import type { HarnessModel } from "./contract"

export const modelInput = (model: HarnessModel): ReadonlyArray<"text" | "image"> =>
  model.capabilities.vision ? ["text", "image"] : ["text"]

export const modelMaxTokens = (model: HarnessModel): number => Math.min(model.contextWindow, 32_768)

export const zeroCost = () => ({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 })
