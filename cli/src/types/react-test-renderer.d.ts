declare module 'react-test-renderer' {
  import type { ReactNode } from 'react'

  export interface ReactTestInstance {
    readonly type: unknown
    readonly props: Record<string, unknown>
    readonly children: readonly (ReactTestInstance | string)[]
    findAll(predicate: (node: ReactTestInstance) => boolean): ReactTestInstance[]
    findByType(type: unknown): ReactTestInstance
    findAllByType(type: unknown): ReactTestInstance[]
  }

  export interface ReactTestRendererJSON {
    type: string
    props: { [propName: string]: unknown }
    children: null | Array<ReactTestRendererJSON | string>
  }

  export interface ReactTestRenderer {
    toJSON(): ReactTestRendererJSON | Array<ReactTestRendererJSON> | null
    unmount(): void
    update(element: ReactNode): void
    root: ReactTestInstance
  }

  export function create(
    nextElement: ReactNode,
    options?: Record<string, unknown>,
  ): ReactTestRenderer

  export function act(callback: () => void | Promise<void>): Promise<void>
}
