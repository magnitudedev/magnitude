/**
 * Terminal background detection — CLI-level one-shot.
 *
 * Detects the terminal's actual background color via OSC query and writes
 * it to `themeAtom.terminalDetectedBg`. Used by floating overlays (dropdowns,
 * menus) as a solid background that matches the terminal — clears the area
 * without looking like a different color.
 *
 * No useEffect: one-shot work is guarded by a ref (spec §5.6 rule 11).
 */
import { useRef } from 'react'
import { useAtomSet } from '@effect-atom/atom-react'
import { useRenderer } from '@opentui/react'
import { themeAtom } from './use-theme'

export function useTerminalBgDetection(): void {
  const renderer = useRenderer()
  const setTheme = useAtomSet(themeAtom)

  const detectedRef = useRef(false)
  if (!detectedRef.current) {
    detectedRef.current = true
    renderer.getPalette({ timeout: 1000 }).then((colors) => {
      const bg = colors?.defaultBackground
      if (bg) {
        setTheme((prev): typeof prev => ({ ...prev, terminalDetectedBg: bg }))
      }
    }).catch(() => {})
  }
}
