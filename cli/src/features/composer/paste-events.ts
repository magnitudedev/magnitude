import { decodePasteBytes } from '@opentui/core'

export function decodeNativePasteText(event: { bytes: Uint8Array }): string {
  return decodePasteBytes(event.bytes)
}
