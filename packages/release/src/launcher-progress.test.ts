import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import type { ArtifactInstallationEvent } from "./installation-progress"
import { makeLauncherInstallationProgress } from "./launcher-progress"

const download = (
  acceptedBytes: number,
  totalBytes = 10 * 1024 ** 2,
): ArtifactInstallationEvent => ({
  _tag: "Downloading",
  progress: {
    strategy: "Segmented",
    acceptedBytes,
    totalBytes,
    attempt: 1,
  },
})

describe("launcher installation progress", () => {
  it("updates one terminal line only when the displayed percentage changes", async () => {
    const writes: string[] = []
    const progress = makeLauncherInstallationProgress({
      isTTY: true,
      write: (text) => writes.push(text),
    })

    await Effect.runPromise(progress.observer.report(download(1024)))
    await Effect.runPromise(progress.observer.report(download(2048)))
    await Effect.runPromise(progress.observer.report(download(5 * 1024 ** 2)))
    await Effect.runPromise(progress.succeeded)

    expect(writes).toEqual([
      "\r\u001b[2K⠋ Downloading Magnitude CLI... 0% (1 KB / 10.5 MB)",
      "\r\u001b[2K⠙ Downloading Magnitude CLI... 50% (5.2 MB / 10.5 MB)",
      "\r\u001b[2K",
      "✓ Magnitude CLI installed\n",
    ])
  })

  it("emits stable milestones instead of terminal control codes when redirected", async () => {
    const writes: string[] = []
    const progress = makeLauncherInstallationProgress({
      isTTY: false,
      write: (text) => writes.push(text),
    })

    await Effect.runPromise(progress.observer.report(download(1024)))
    await Effect.runPromise(progress.observer.report(download(2048)))
    await Effect.runPromise(progress.observer.report({
      _tag: "Verifying",
      progress: { completedBytes: 1024, totalBytes: 2048 },
    }))
    await Effect.runPromise(progress.observer.report({
      _tag: "Extracting",
      progress: { completedBytes: 1024, totalBytes: 2048 },
    }))
    await Effect.runPromise(progress.succeeded)

    expect(writes).toEqual([
      "Downloading Magnitude CLI...\n",
      "Verifying Magnitude CLI...\n",
      "Installing Magnitude CLI...\n",
      "✓ Magnitude CLI installed\n",
    ])
  })

  it("replaces download with unmeasured verification and installation phases", async () => {
    const writes: string[] = []
    const progress = makeLauncherInstallationProgress({
      isTTY: true,
      write: (text) => writes.push(text),
    })

    await Effect.runPromise(progress.observer.report(download(5 * 1024 ** 2)))
    await Effect.runPromise(progress.observer.report({
      _tag: "Verifying",
      progress: { completedBytes: 1024, totalBytes: 2048 },
    }))
    await Effect.runPromise(progress.observer.report({
      _tag: "Extracting",
      progress: { completedBytes: 1024, totalBytes: 2048 },
    }))
    await Effect.runPromise(progress.succeeded)

    expect(writes).toEqual([
      "\r\u001b[2K⠋ Downloading Magnitude CLI... 50% (5.2 MB / 10.5 MB)",
      "\r\u001b[2K⠙ Verifying Magnitude CLI...",
      "\r\u001b[2K⠹ Installing Magnitude CLI...",
      "\r\u001b[2K",
      "✓ Magnitude CLI installed\n",
    ])
  })

  it("clears an interactive status on failure without claiming installation", async () => {
    const writes: string[] = []
    const progress = makeLauncherInstallationProgress({
      isTTY: true,
      write: (text) => writes.push(text),
    })

    await Effect.runPromise(progress.observer.report(download(1024)))
    await Effect.runPromise(progress.failed)

    expect(writes.at(-1)).toBe("\r\u001b[2K")
    expect(writes.join("")).not.toContain("installed")
  })
})
