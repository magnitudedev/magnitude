import { createElement, type ReactNode } from "react"
import { act, create, type ReactTestRenderer } from "react-test-renderer"
import { RegistryContext } from "@effect-atom/atom-react"
import * as Registry from "@effect-atom/atom/Registry"
import { describe, expect, it, vi } from "vitest"
import { Composer } from "./composer"

vi.mock("@/components/ui/tooltip", () => ({
  ActionTooltip: ({ trigger }: { readonly trigger: ReactNode }) => trigger,
}))

;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true

describe("composer submission identity", () => {
  it("consumes one draft occurrence at most once", async () => {
    const registry = Registry.make()
    const submissions: string[] = []
    let renderer!: ReactTestRenderer

    await act(async () => {
      renderer = create(createElement(
        RegistryContext.Provider,
        { value: registry },
        createElement(Composer, {
          onSend: (text) => submissions.push(text),
          mentionClient: null,
        }),
      ))
    })

    const textarea = renderer.root.findByType("textarea")
    await act(async () => {
      textarea.props.onChange({
        target: { value: "one draft", style: {}, scrollHeight: 0 },
      })
    })

    const readyTextarea = renderer.root.findByType("textarea")
    const nativeEvent = {
      key: "Enter",
      shiftKey: false,
      ctrlKey: false,
      metaKey: false,
      altKey: false,
      defaultPrevented: false,
    }
    const event = {
      nativeEvent,
      key: "Enter",
      shiftKey: false,
      metaKey: false,
      ctrlKey: false,
      currentTarget: { selectionStart: "one draft".length },
      preventDefault: () => undefined,
    }

    await act(async () => {
      readyTextarea.props.onKeyDown(event)
      readyTextarea.props.onKeyDown(event)
    })

    expect(submissions).toEqual(["one draft"])

    await act(async () => renderer.unmount())
    registry.dispose()
  })
})
