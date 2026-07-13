export type ActionId = 'open-settings' | 'open-usage'

export type ErrorCta =
  | { readonly kind: 'url'; readonly label: string; readonly url: string }
  | { readonly kind: 'action'; readonly label: string; readonly actionId: string; readonly chord: string }
