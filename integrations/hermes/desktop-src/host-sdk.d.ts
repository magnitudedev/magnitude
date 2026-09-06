// The native disk-plugin SDK is supplied by Hermes, not published to npm.
// This declaration describes only the public surface this companion consumes.
// The packed companion is also loaded through Hermes's real runtime in acceptance.
declare module "@hermes/plugin-sdk" {
  import type { ComponentType, ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react"
  export { useQuery, useMutation } from "@tanstack/react-query"

  interface Readable<A> { get(): A }
  export function useValue<A>(value: Readable<A>): A
  export const Button: ComponentType<ButtonHTMLAttributes<HTMLButtonElement>>
  export const Input: ComponentType<InputHTMLAttributes<HTMLInputElement>>
  export const STATUSBAR_AREAS: { readonly left: "statusBar.left"; readonly right: "statusBar.right" }
  export const PALETTE_AREA: string
  export const host: {
    readonly state: {
      readonly connectionId: Readable<string | null>
      readonly profile: Readable<string>
      readonly gateway: Readable<string>
      readonly busy: Readable<boolean>
      readonly focusedStoredSessionId: Readable<string | null>
      readonly focusedSessionOwner: Readable<{ connectionId: string; profile: string } | null>
    }
    openWorkspace(id: string, options: { title: string; render: () => ReactNode }): () => void
  }
  interface Contribution {
    readonly id: string
    readonly area: string
    readonly order?: number
    readonly render?: () => ReactNode
    readonly data?: unknown
  }
  export interface PluginContext {
    readonly source: string
    rest<A>(path: string, options?: { method?: string; body?: unknown; timeoutMs?: number }): Promise<A>
    registerMany(contributions: Contribution[]): () => void
    onDispose(dispose: () => void): void
  }
}
