import { describe, expect, test, vi } from "vitest"
import { connectCloudAndFinish, finishModelSetup } from "./screen"

describe("combined model setup flow", () => {
  test("connects Cloud before completing the first-run walkthrough", async () => {
    const order: string[] = []
    const configureCloud = vi.fn(async (key: string) => {
      order.push(`cloud:${key}`)
    })
    const completeOnboarding = vi.fn(async () => {
      order.push("onboarding")
      return true
    })
    const onComplete = vi.fn(() => order.push("done"))

    await connectCloudAndFinish(
      "cloud-key",
      configureCloud,
      () => finishModelSetup("onboarding", completeOnboarding, onComplete),
    )
    expect(order).toEqual(["cloud:cloud-key", "onboarding", "done"])
  })

  test("allows skipping Cloud while still completing first-run onboarding", async () => {
    const completeOnboarding = vi.fn(async () => true)
    const onComplete = vi.fn()
    expect(await finishModelSetup("onboarding", completeOnboarding, onComplete)).toBe(true)
    expect(completeOnboarding).toHaveBeenCalledOnce()
    expect(onComplete).toHaveBeenCalledOnce()
  })

  test("returns to Settings without rewriting onboarding in management mode", async () => {
    const completeOnboarding = vi.fn(async () => true)
    const onComplete = vi.fn()
    expect(await finishModelSetup("management", completeOnboarding, onComplete)).toBe(true)
    expect(completeOnboarding).not.toHaveBeenCalled()
    expect(onComplete).toHaveBeenCalledOnce()
  })

  test("does not leave onboarding when persistence reports failure", async () => {
    const onComplete = vi.fn()
    expect(await finishModelSetup("onboarding", async () => false, onComplete)).toBe(false)
    expect(onComplete).not.toHaveBeenCalled()
  })
})
