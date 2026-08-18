import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"

import { clampResizableValue } from "@/lib/resizable"
import { ResizableEdge } from "./resizable-edge"

describe("ResizableEdge", () => {
  it("exposes an accessible keyboard-focusable separator", () => {
    const html = renderToStaticMarkup(
      <ResizableEdge
        side="left"
        value={320}
        minimum={280}
        maximum={800}
        onValueChange={() => undefined}
        label="Resize project files"
      />,
    )

    expect(html).toContain('role="separator"')
    expect(html).toContain('aria-label="Resize project files"')
    expect(html).toContain('aria-orientation="vertical"')
    expect(html).toContain('aria-valuemin="280"')
    expect(html).toContain('aria-valuemax="800"')
    expect(html).toContain('aria-valuenow="320"')
    expect(html).toContain('tabindex="0"')
  })

  it("rounds and clamps resize values", () => {
    expect(clampResizableValue(319.6, 280, 800)).toBe(320)
    expect(clampResizableValue(200, 280, 800)).toBe(280)
    expect(clampResizableValue(900, 280, 800)).toBe(800)
  })
})
