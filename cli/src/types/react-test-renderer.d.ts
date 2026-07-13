declare module 'react-test-renderer' {
  import type { ReactElement } from 'react'

  export interface ReactTestRendererJSON {
    type: string
    props: { [propName: string]: unknown }
    children: null | Array<ReactTestRendererJSON | string>
  }

  export interface ReactTestRenderer {
    toJSON(): ReactTestRendererJSON | Array<ReactTestRendererJSON> | null
    unmount(): void
    update(element: ReactElement): void
    root: { findByType: (type: unknown) => unknown; findAllByType: (type: unknown) => unknown[] }
  }

  export function create(
    nextElement: ReactElement,
    options?: Record<string, unknown>,
  ): ReactTestRenderer

  export function act(callback: () => void | Promise<void>): Promise<void>
}
