import { act, useState } from "react"
import { testRender } from "@opentui/react/test-utils"
import { Option } from "effect"
import { expect, test, vi } from "vitest"
import { defaultCliThemes } from "../utils/theme"
import {
  PENTAGON_RADAR_DURATION_MS,
  renderPentagonRadar,
  type PentagonRadarAxes,
} from "./pentagon-radar"

const clock = vi.hoisted(() => ({
  time: 0,
  listeners: new Set<() => void>(),
}))

vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@magnitudedev/client-common")>()),
  getAnimationTimeSnapshot: () => clock.time,
  subscribeAnimationClock: (listener: () => void) => {
    clock.listeners.add(listener)
    return () => clock.listeners.delete(listener)
  },
}))

vi.mock("../hooks/use-theme", () => ({
  useTheme: () => defaultCliThemes.dark,
}))

const { PentagonRadarView } = await import("./pentagon-radar-view")

const axes = (value: number): PentagonRadarAxes => [
  { value: Option.some(value), label: "A", detail: "1" },
  { value: Option.some(value), label: "B", detail: "2" },
  { value: Option.some(value), label: "C", detail: "3" },
  { value: Option.some(value), label: "D", detail: "4" },
  { value: Option.some(value), label: "E", detail: "5" },
]

const renderedFrame = (value: PentagonRadarAxes): string => renderPentagonRadar(value)
  .map((row) => row.map(({ text }) => text).join(""))
  .join("\n")
  .trimEnd()

test("automatically transitions when passed radar values change", async () => {
  clock.time = 0
  clock.listeners.clear()
  let updateAxes: ((next: PentagonRadarAxes) => void) | undefined
  const initial = axes(0)
  const target = axes(1)
  const StatefulRadar = () => {
    const [value, setValue] = useState(initial)
    updateAxes = setValue
    return <PentagonRadarView axes={value} />
  }
  const view = await testRender(<StatefulRadar />, { width: 56, height: 15 })

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame().trimEnd()).toBe(renderedFrame(initial))

    clock.time = 100
    await act(async () => {
      updateAxes?.(target)
      await view.renderOnce()
    })
    expect(view.captureCharFrame().trimEnd()).toBe(renderedFrame(initial))

    clock.time = 100 + PENTAGON_RADAR_DURATION_MS / 2
    await act(async () => {
      clock.listeners.forEach((listener) => listener())
      await view.renderOnce()
    })
    const midpoint = view.captureCharFrame().trimEnd()
    expect(midpoint).not.toBe(renderedFrame(initial))
    expect(midpoint).not.toBe(renderedFrame(target))

    clock.time = 100 + PENTAGON_RADAR_DURATION_MS
    await act(async () => {
      clock.listeners.forEach((listener) => listener())
      await view.renderOnce()
    })
    expect(view.captureCharFrame().trimEnd()).toBe(renderedFrame(target))
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
