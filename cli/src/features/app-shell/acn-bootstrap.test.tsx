import { act, useState } from "react"
import { KeyEvent } from "@opentui/core"
import { testRender } from "@opentui/react/test-utils"
import { Option } from "effect"
import type { AcnStartingPhase } from "@magnitudedev/sdk"
import { expect, test, vi } from "vitest"
import { defaultCliThemes } from "../../utils/theme"

const keyboard = vi.hoisted(
  (): { handler: ((key: KeyEvent) => void) | undefined } => ({
    handler: undefined,
  }),
)

vi.mock("@opentui/react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@opentui/react")>()),
  useKeyboard: (handler: (key: KeyEvent) => void) => {
    keyboard.handler = handler
  },
}))

vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => defaultCliThemes.dark,
}))

const { AcnBootstrapScreen } = await import("./acn-bootstrap")

test.each([
  ["PreparingAcn", "Preparing background server"],
  ["WaitingForOwner", "Waiting for previous Magnitude process"],
  ["ResolvingLocalInference", "Preparing local inference"],
  ["LaunchingLocalInference", "Starting local inference"],
] as const)("renders the %s startup phase", async (phase, label) => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{ _tag: "Starting", phase }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain(label)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("renders CUDA backend preparation with its hardware label", async () => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{
        _tag: "Starting",
        phase: {
          _tag: "PreparingBackend",
          backend: { _tag: "Cuda", hardwareLabel: "NVIDIA GeForce RTX 3060" },
        },
      }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain(
      "Preparing CUDA backend for NVIDIA GeForce RTX 3060",
    )
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("renders Metal backend preparation on a Mac", async () => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{
        _tag: "Starting",
        phase: {
          _tag: "PreparingBackend",
          backend: { _tag: "Metal", hardwareLabel: "Apple M4 Max" },
        },
      }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain(
      "Preparing Metal backend for Apple M4 Max",
    )
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

const keyEvent = (name: string, ctrl = false) =>
  new KeyEvent({
    name,
    ctrl,
    meta: false,
    shift: false,
    option: false,
    sequence: "",
    number: false,
    raw: "",
    eventType: "press",
    source: "raw",
  })

test("renders exactly one empty row between the installation title and bar", async () => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{
        _tag: "Installing",
        phase: "DownloadingInferenceEngine",
        overallProgress: 0.88,
        detailIsExact: true,
        detail: Option.some({
          completed: 19 * 1024 * 1024,
          totalBytes: 19.7 * 1024 * 1024,
          unit: "Bytes",
          attempt: Option.some(1),
        }),
      }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    const lines = view.captureCharFrame().split("\n")
    const titleRow = lines.findIndex((line) =>
      line.includes("Installing Magnitude"),
    )
    const barRow = lines.findIndex((line) => line.includes("88%"))

    expect(barRow - titleRow).toBe(2)
    expect(lines[titleRow + 1]?.trim()).toBe("")
    expect(view.captureCharFrame()).toContain("19.9 MB of 20.7 MB")
    expect(view.captureCharFrame()).not.toMatch(/[\u2800-\u28ff]/u)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("centers the Starting title, radar loader, and subtitle as one stack", async () => {
  const width = 91
  const height = 13
  const subtitle = "Preparing local inference"
  const view = await testRender(
    <AcnBootstrapScreen
      state={{ _tag: "Starting", phase: "ResolvingLocalInference" }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width, height },
  )

  try {
    await act(view.renderOnce)
    const lines = view.captureCharFrame().split("\n")
    const titleRow = lines.findIndex((line) => line.includes("Starting Magnitude"))
    const subtitleRow = lines.findIndex((line) => line.includes(subtitle))
    const terminalCenter = (width - 1) / 2
    const textCenter = (line: string, text: string): number =>
      line.indexOf(text) + (text.length - 1) / 2

    expect(subtitleRow - titleRow).toBe(7)
    expect(Math.abs(textCenter(lines[titleRow]!, "Starting Magnitude") - terminalCenter))
      .toBeLessThanOrEqual(0.5)
    expect(Math.abs(textCenter(lines[subtitleRow]!, subtitle) - terminalCenter))
      .toBeLessThanOrEqual(0.5)
    expect(Math.abs((titleRow + subtitleRow) / 2 - (height - 1) / 2)).toBeLessThanOrEqual(1)

    const radarRows = lines.slice(titleRow + 1, subtitleRow)
    expect(radarRows).toHaveLength(6)
    expect(radarRows.some((line) => /[\u2800-\u28ff]/u.test(line))).toBe(true)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("keeps one radar loader mounted across Starting phase changes", async () => {
  let setPhase: ((phase: AcnStartingPhase) => void) | undefined
  const StatefulBootstrap = () => {
    const [phase, updatePhase] = useState<AcnStartingPhase>("PreparingAcn")
    setPhase = (next) => updatePhase(next)
    return (
      <AcnBootstrapScreen
        state={{ _tag: "Starting", phase }}
        onRetry={() => undefined}
        onQuit={() => undefined}
      />
    )
  }
  const random = vi.spyOn(Math, "random").mockReturnValue(0.5)
  const view = await testRender(<StatefulBootstrap />, { width: 91, height: 13 })

  try {
    await act(view.renderOnce)
    expect(random).toHaveBeenCalledOnce()
    await act(async () => setPhase?.("LaunchingLocalInference"))
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Starting local inference")
    expect(random).toHaveBeenCalledOnce()
  } finally {
    random.mockRestore()
    await act(async () => view.renderer.destroy())
  }
})

test("renders a static Failed state without a radar loader", async () => {
  const view = await testRender(
    <AcnBootstrapScreen
      state={{ _tag: "Failed", stage: "LaunchDaemon", message: "boom", retryable: true }}
      onRetry={() => undefined}
      onQuit={() => undefined}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    expect(view.captureCharFrame()).toContain("Magnitude failed to start")
    expect(view.captureCharFrame()).not.toMatch(/[\u2800-\u28ff]/u)
  } finally {
    await act(async () => view.renderer.destroy())
  }
})

test("quits on Ctrl-C during startup without consuming unrelated keys", async () => {
  const onQuit = vi.fn()
  const view = await testRender(
    <AcnBootstrapScreen
      state={{ _tag: "Starting", phase: "PreparingAcn" }}
      onRetry={() => undefined}
      onQuit={onQuit}
    />,
    { width: 91, height: 13 },
  )

  try {
    await act(view.renderOnce)
    const unrelated = keyEvent("x")
    keyboard.handler?.(unrelated)
    expect(unrelated.defaultPrevented).toBe(false)

    const ctrlC = keyEvent("c", true)
    keyboard.handler?.(ctrlC)
    expect(ctrlC.defaultPrevented).toBe(true)
    expect(onQuit).toHaveBeenCalledOnce()
  } finally {
    await act(async () => view.renderer.destroy())
  }
})
