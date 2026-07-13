import stringWidth from 'string-width'

/**
 * Width-aware item fitting: fits as many string items as possible within
 * `maxWidth`, reserving room for a trailing `, +N more` suffix when items
 * remain. Pure data transform — no React, no theme, no surface dependency.
 */
export function fitItems(items: string[], maxWidth: number): { shown: string[]; remaining: number } {
  if (items.length === 0) return { shown: [], remaining: 0 }

  const shown: string[] = []
  let used = 0
  let i = 0

  while (i < items.length) {
    const item = items[i]
    const itemWidth = stringWidth(item)
    const sepWidth = shown.length > 0 ? 2 : 0
    // Always include ', ' before suffix — by render time, shown.length > 0
    const suffixWidth = i < items.length - 1 ? stringWidth(`, +${items.length - i - 1} more`) : 0
    const available = maxWidth - used - sepWidth - suffixWidth

    if (itemWidth <= available) {
      shown.push(item)
      used += sepWidth + itemWidth
      i++
    } else {
      // Item doesn't fit fully — stop here, put this and the rest behind "+N more"
      break
    }
  }

  return { shown, remaining: items.length - i }
}
