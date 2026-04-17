import { catalog } from '../catalog'

export type ToolState = {
  [K in keyof typeof catalog.entries]:
    (typeof catalog.entries)[K] extends { state: { initial: infer S } }
      ? S
      : never
}[keyof typeof catalog.entries]
