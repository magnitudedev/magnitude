import { CHARS_PER_TOKEN_LOWER } from '../constants'
import type { UserPart } from '@magnitudedev/ai'
import { renderContextImageAnchor, type ContextPart } from '../content'

/**
 * Kimi K2.6 image token estimation.
 * Derived from MoonViT config: patch_size=14, merge_kernel_size=[2,2]
 */
export const DEFAULT_IMAGE_TOKENS = 1000 // fallback when dimensions unknown

export function estimateImageTokens(width: number | null, height: number | null): number {
  if (width == null || height == null) return DEFAULT_IMAGE_TOKENS
  const mergedH = Math.ceil(Math.ceil(height / 14) / 2)
  const mergedW = Math.ceil(Math.ceil(width / 14) / 2)
  return mergedH * mergedW
}

export function estimateText(s: string | undefined): number {
  if (!s) return 0
  return Math.ceil(s.length / CHARS_PER_TOKEN_LOWER)
}

export function estimateContentTokens(content: string): number
export function estimateContentTokens(content: readonly (UserPart | ContextPart)[]): number
export function estimateContentTokens(content: string | readonly (UserPart | ContextPart)[]): number {
  if (typeof content === 'string') {
    return Math.ceil(content.length / CHARS_PER_TOKEN_LOWER)
  }
  let tokens = 0
  for (const part of content) {
    switch (part._tag) {
      case 'TextPart':
      case 'ContextText':
        tokens += Math.ceil(part.text.length / CHARS_PER_TOKEN_LOWER)
        break
      case 'ImagePart':
        tokens += part.dimensions
          ? estimateImageTokens(part.dimensions.width, part.dimensions.height)
          : DEFAULT_IMAGE_TOKENS
        break
      case 'ContextImage':
        tokens += estimateText(renderContextImageAnchor(part))
        tokens += estimateImageTokens(part.dimensions.width, part.dimensions.height)
        break
    }
  }
  return tokens
}
