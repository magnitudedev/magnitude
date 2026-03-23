
import React from 'react'
import { describe, expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

let capturedScrollboxOnPaste:
  | ((event: { bytes: Uint8Array }) => void)
  | undefined

mock.module('react/jsx-runtime', () => ({
  Fragment: React.Fragment,
  jsx: (type: unknown, props: Record<string, unknown>, key?: string) => {
    if (type === 'scrollbox') {
      capturedScrollboxOnPaste = props.onPaste as
        | ((event: { bytes: Uint8Array }) => void)
        | undefined
    }
    return React.createElement(type as any, { ...props, key })
  },
  jsxs: (type: unknown, props: Record<string, unknown>, key?: string) => {
    if (type === 'scrollbox') {
      capturedScrollboxOnPaste = props.onPaste as
        | ((event: { bytes: Uint8Array }) => void)
        | undefined
    }
    return React.createElement(type as any, { ...props, key })
  },
}))

mock.module('react/jsx-dev-runtime', () => ({
  Fragment: React.Fragment,
  jsxDEV: (
    type: unknown,
    props: Record<string, unknown>,
    key?: string,
  ) => {
    if (type === 'scrollbox') {
      capturedScrollboxOnPaste = props.onPaste as
        | ((event: { bytes: Uint8Array }) => void)
        | undefined
    }
    return React.createElement(type as any, { ...props, key })
  },
}))

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    muted: 'gray',
    inputFocusedFg: 'white',
    inputFg: 'white',
    info: 'blue',
    primary: 'cyan',
    link: 'magenta',
  }),
}))

mock.module('../hooks/use-mounted-ref', () => ({
  useMountedRef: () => ({ current: true }),
}))

mock.module('../hooks/use-safe-event', () => ({
  useSafeEvent: (fn: unknown) => fn,
}))

mock.module('../hooks/use-safe-interval', () => ({
  useSafeInterval: () => ({ set: setInterval, clear: clearInterval }),
}))

mock.module('../hooks/use-safe-timeout', () => ({
  useSafeTimeout: () => ({ set: setTimeout, clear: clearTimeout }),
}))

mock.module('../utils/theme', () => ({
  terminalSupportsRgb24: () => false,
  orange: { 400: 'darkyellow' },
}))

mock.module('@opentui/react', () => ({
  useKeyboard: () => {},
  useRenderer: () => ({ clearSelection() {} }),
}))

mock.module('@opentui/core', () => ({
  TextAttributes: { BOLD: 1 },
  RGBA: { fromInts: () => 'rgba(0,0,0,0)' },
  decodePasteBytes: (bytes: Uint8Array) => new TextDecoder().decode(bytes),
}))

const { MultilineInput } = await import('./multiline-input')

describe('MultilineInput native paste integration', () => {
  test('native bytes paste decodes and forwards text to shared paste handler callback', () => {
    capturedScrollboxOnPaste = undefined
    const onPasteCalls: Array<string | undefined> = []

    renderToStaticMarkup(
      <MultilineInput
        value=""
        cursorPosition={0}
        onChange={() => {}}
        onSubmit={() => {}}
        onPaste={(text) => {
        onPasteCalls.push(text)
        return Boolean(text)
      }}
      />,
    )

    expect(typeof capturedScrollboxOnPaste).toBe('function')

    capturedScrollboxOnPaste?.({ bytes: new Uint8Array([65, 66, 67]) })

    expect(onPasteCalls).toEqual(['ABC'])
  })
})
