import { renderToStaticMarkup } from "react-dom/server"
import { expect, test, vi } from "vitest"

vi.mock("@opentui/react", () => ({
  useKeyboard: () => {},
  useRenderer: () => ({ setMousePointer: () => {}, requestRender: () => {} }),
}))
vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
    success: "green",
    error: "red",
    warning: "yellow",
    border: "gray",
    terminalDetectedBg: "black",
  }),
}))
vi.mock("../../components/button", () => ({
  Button: ({ children }: { children?: unknown }) => <button>{children as never}</button>,
}))

const { SettingsOverlay } = await import("./settings")

test("settings exposes direct Local Models and Cloud Fallback re-entry", () => {
  const html = renderToStaticMarkup(
    <SettingsOverlay
      isVisible
      onClose={() => {}}
      auth={{
        source: "none",
        key: null,
        maskedKey: null,
        envVarName: null,
        save: async () => {},
        clear: async () => {},
      }}
      slots={[]}
      onManageLocalModels={() => {}}
      onConfigureCloud={() => {}}
    />,
  )
  expect(html).toContain("Inference sources")
  expect(html).toContain("Manage local models")
  expect(html).toContain("Configure Cloud fallback")
})
