import { renderToStaticMarkup } from "react-dom/server"
import { createElement } from "react"
import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelSlotConfiguredLocal,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type DisplayActor,
  type DisplayRootStatus,
} from "@magnitudedev/sdk"
import { isWorkStatusBarVisible, WorkStatusBar } from "./work-status-bar"

const worked: DisplayRootStatus = {
  _tag: "Worked",
  lastProductiveMs: 5_000,
}

const working: DisplayRootStatus = {
  _tag: "Working",
  chainStartedAt: 1,
  detail: { _tag: "NoDetail" },
  activeChildCount: 0,
}

const rootActor: DisplayActor = {
  kind: "root",
  name: "Leader",
  role: "leader",
  parentActorKey: null,
  taskId: null,
  status: {
    ...working,
    detail: { _tag: "WaitingForModel", turnStartedAt: 1 },
  },
  context: { tokenEstimate: 0, isCompacting: false },
}

const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("test-configuration"),
  reasoningEffort: ReasoningEffortSchema.make("none"),
}

const requestedModel = new ModelSlotConfiguredLocal({
  slotId: PRIMARY_SLOT_ID,
  selection,
  descriptor: {
    providerId: selection.providerId,
    providerModelId: selection.providerModelId,
    displayName: "Local test",
    variantLabel: Option.none(),
  },
  availability: { _tag: "Available" },
  residency: { _tag: "Requested" },
  actions: ["Stop"],
})

describe("isWorkStatusBarVisible", () => {
  it("shows live work above the composer", () => {
    expect(isWorkStatusBarVisible(working, false)).toBe(true)
  })

  it("moves completed-work presentation into the timeline", () => {
    expect(isWorkStatusBarVisible(worked, false)).toBe(false)
  })

  it("keeps the task panel available after root work completes", () => {
    expect(isWorkStatusBarVisible(worked, true)).toBe(true)
  })

  it("shows requested model acquisition as loading instead of waiting", () => {
    const markup = renderToStaticMarkup(createElement(WorkStatusBar, {
      rootActor,
      actors: { root: rootActor },
      tasks: null,
      modelLoadActivity: requestedModel,
    }))

    expect(markup).toContain("Loading model")
    expect(markup).toContain("· 0%")
    expect(markup).not.toContain("Waiting for model")
    expect(markup).toContain('role="status"')
  })
})
