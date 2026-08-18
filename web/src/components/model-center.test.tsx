import { Result } from "@effect-atom/atom-react"
import { renderToStaticMarkup } from "react-dom/server"
import { beforeEach, describe, expect, it, vi } from "vitest"

const modelState = vi.hoisted(
  (): {
    localModels: unknown
    slots: unknown
    catalog: unknown
  } => ({
    localModels: undefined,
    slots: undefined,
    catalog: undefined,
  })
)

vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@magnitudedev/client-common")>()),
  useLocalModels: () => modelState.localModels,
  useModelSlots: () => modelState.slots,
  useCatalogModels: () => modelState.catalog,
  useLocalModelActions: () => ({ latestInstallationFailed: false }),
  usePlatform: () => ({
    id: "desktop",
    showItemInFolder: vi.fn(),
  }),
}))

import { SettingsCenter } from "./model-center"

beforeEach(() => {
  modelState.localModels = Result.initial(false)
  modelState.slots = Result.initial(false)
  modelState.catalog = Result.initial(false)
})

describe("Settings query states", () => {
  it("renders the initial query state as loading before a request starts", () => {
    const html = renderToStaticMarkup(<SettingsCenter tab="catalog" />)

    expect(html).toContain("Loading catalog")
    expect(html).toContain("Checking the model catalog for this computer.")
    expect(html).toContain("animate-spin")
  })
})

describe("Settings model loading states", () => {
  it("does not render an initializing model inventory as an empty library", () => {
    modelState.localModels = Result.success({
      inventoryState: { _tag: "Initializing" },
      discoveryState: { _tag: "Loading", progress: [] },
      models: [],
    })

    const html = renderToStaticMarkup(<SettingsCenter tab="models" />)

    expect(html).toContain("Loading models")
    expect(html).toContain("Reading the models installed on this computer.")
    expect(html).not.toContain("No models are installed yet")
    expect(html).not.toContain("Configured models")
  })

  it("renders Models as a searchable installed-model library only", () => {
    modelState.localModels = Result.success({
      inventoryState: { _tag: "Ready" },
      discoveryState: { _tag: "Ready", progress: [] },
      models: [],
    })

    const html = renderToStaticMarkup(<SettingsCenter tab="models" />)

    expect(html).toContain("Installed models")
    expect(html).toContain('aria-label="Search installed models"')
    expect(html).toContain("No models are installed yet")
    expect(html).not.toContain("Configured models")
    expect(html).not.toContain("Primary")
    expect(html).not.toContain("Secondary")
    expect(html).not.toContain("residency")
  })

  it("shows catalog discovery as loading instead of an empty candidate layout", () => {
    modelState.catalog = Result.success({
      discoveryState: { _tag: "Loading", progress: [] },
      models: [],
    })

    const html = renderToStaticMarkup(<SettingsCenter tab="catalog" />)

    expect(html).toContain("Loading catalog")
    expect(html).toContain("Assessing local models for this computer.")
    expect(html).not.toContain('aria-label="Catalog models"')
    expect(html).not.toContain(
      "No local catalog models are currently available"
    )
  })

  it("renders the catalog empty state only after discovery is ready", () => {
    modelState.catalog = Result.success({
      discoveryState: { _tag: "Ready", progress: [] },
      models: [],
    })

    const html = renderToStaticMarkup(<SettingsCenter tab="catalog" />)

    expect(html).toContain("No local catalog models are currently available.")
    expect(html).toContain('aria-label="Search catalog"')
    expect(html).toContain('aria-label="Filter catalog"')
    expect(html).toContain('aria-label="Sort catalog"')
    expect(html).toContain("Installed")
    expect(html).not.toContain("Loading catalog")
  })

  it("does not turn failed catalog discovery into a successful empty state", () => {
    modelState.catalog = Result.success({
      discoveryState: {
        _tag: "Failed",
        progress: [],
        failure: { message: "Assessment failed." },
      },
      models: [],
    })

    const html = renderToStaticMarkup(<SettingsCenter tab="catalog" />)

    expect(html).toContain("Assessment failed.")
    expect(html).not.toContain("No local catalog models are currently available.")
  })
})
