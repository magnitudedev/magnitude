const encodePart = (part: string): string =>
  encodeURIComponent(part)

export const joinAddress = (parts: Iterable<string>): string =>
  [...parts].map(encodePart).join('/')

export const childAddress = (prefix: string, ...parts: readonly string[]): string =>
  [prefix, ...parts.map(encodePart)].join('/')

/**
 * Sentinel address for a collection instance's index structure. Marked as
 * changed by structural operations (append, insert, remove, member add/remove)
 * and recorded by consumer proxies on any access, so structural changes reach
 * consumers even when no existing entry content was rewritten. Address parts
 * are URI-encoded, so a raw `#` suffix can never collide with an entry address.
 */
export const collectionSentinelAddress = (prefix: string): string => `${prefix}#index`

export const isCollectionSentinelAddress = (address: string): boolean =>
  address.endsWith('#index')
