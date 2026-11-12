import { renderToStaticMarkup } from "react-dom/server"
import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { SidebarUpdatePill, sidebarUpdate, UpdatePill, type SidebarUpdate, type UpdateTransfer } from "./application-update"

const noop = () => {}
const render = (update: SidebarUpdate) => renderToStaticMarkup(<UpdatePill update={update} busy={false} onDownload={noop} onRestart={noop} onOpenSettings={noop} />)

describe("sidebar update", () => {
  it("offers nothing while up to date, checking fails, or updates can't happen here", () => {
    const quiet: Array<UpdateTransfer | undefined> = [undefined, { _tag: "Idle" }, { _tag: "Cancelling" }, { _tag: "Failed", message: "offline" }, { _tag: "Unavailable", message: "dev build" }, { _tag: "Closed" }]
    for (const transfer of quiet) {
      expect(sidebarUpdate(transfer)).toEqual({ _tag: "None" })
      expect(render(sidebarUpdate(transfer))).toBe("")
    }
  })

  it("offers a download, and sends a version that failed here to Settings for the explanation", () => {
    expect(sidebarUpdate({ _tag: "Available", version: "0.2.0", bytes: 100, failed: Option.none() })).toEqual({ _tag: "Available", version: "0.2.0", previouslyFailed: false })
    expect(sidebarUpdate({ _tag: "Available", version: "0.2.0", bytes: 100, failed: Option.some("verify") })).toEqual({ _tag: "Available", version: "0.2.0", previouslyFailed: true })
    expect(render({ _tag: "Available", version: "0.2.0", previouslyFailed: false })).toContain("Update available")
    expect(render({ _tag: "Available", version: "0.2.0", previouslyFailed: false })).toContain("Click to download 0.2.0")
    expect(render({ _tag: "Available", version: "0.2.0", previouslyFailed: true })).toContain("Needs attention")
  })

  it("shows download progress without an action", () => {
    expect(sidebarUpdate({ _tag: "Downloading", version: "0.2.0", completed: 25, total: 100 })).toEqual({ _tag: "Downloading", version: "0.2.0", fraction: Option.some(0.25) })
    expect(sidebarUpdate({ _tag: "Downloading", version: "0.2.0", completed: 0, total: 0 })).toEqual({ _tag: "Downloading", version: "0.2.0", fraction: Option.none() })
    expect(sidebarUpdate({ _tag: "Staging", version: "0.2.0" })).toEqual({ _tag: "Preparing", version: "0.2.0" })
    expect(render({ _tag: "Preparing", version: "0.2.0" })).toContain("Preparing update")
    const html = render({ _tag: "Downloading", version: "0.2.0", fraction: Option.some(0.25) })
    expect(html).toContain("25%")
    expect(html).not.toContain("<button")
  })

  it("asks to restart once prepared, and to retry after an installation failed", () => {
    expect(sidebarUpdate({ _tag: "Ready", version: "0.2.0" })).toEqual({ _tag: "Ready", version: "0.2.0", retry: false })
    expect(sidebarUpdate({ _tag: "InstallationFailed", version: "0.2.0", message: "x" })).toEqual({ _tag: "Ready", version: "0.2.0", retry: true })
    expect(render({ _tag: "Ready", version: "0.2.0", retry: false })).toContain("Update 0.2.0 ready")
    expect(render({ _tag: "Ready", version: "0.2.0", retry: true })).toContain("0.2.0 didn’t install")
  })

  it("points a container's new image to Settings", () => {
    expect(sidebarUpdate({ _tag: "ImageAvailable", version: "0.2.0", image: "magnitude:0.2.0" })).toEqual({ _tag: "ImageAvailable", version: "0.2.0" })
    expect(render({ _tag: "ImageAvailable", version: "0.2.0" })).toContain("Update available")
  })

  it("disables its action while a request is pending", () => {
    const html = renderToStaticMarkup(<UpdatePill update={{ _tag: "Ready", version: "0.2.0", retry: false }} busy onDownload={noop} onRestart={noop} onOpenSettings={noop} />)
    expect(html).toMatch(/<button[^>]*disabled=""/)
  })

  it("leaves the boot skeleton's sidebar, which renders before any client exists, without an update row", () => {
    expect(renderToStaticMarkup(<SidebarUpdatePill />)).toBe("")
  })
})
