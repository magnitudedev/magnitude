import { useCallback, useRef, useState } from 'react'

/**
 * Hook that measures the actual rendered width of a box element via ref + onSizeChange.
 * 
 * OpenTUI runs Yoga layout synchronously before painting, so onSizeChange fires
 * before the user sees anything — there is no visible "flash" of wrong width.
 * 
 * Usage:
 *   const box = useBoxWidth()
 *   return <box ref={box.ref} onSizeChange={box.onSizeChange}>
 *     {box.width != null && <MyContent width={box.width} />}
 *   </box>
 * 
 * width is null until first measurement, then the exact Yoga-computed width minus 1 buffer.
 */
export function useBoxWidth() {
  const ref = useRef<any>(null)
  const [width, setWidth] = useState<number | null>(null)

  const onSizeChange = useCallback(() => {
    const w = ref.current?.width
    if (typeof w === 'number' && w > 0) {
      // Subtract 1 as safety buffer for rounding in layout engine
      setWidth(w - 1)
    }
  }, [])

  return { ref, onSizeChange, width }
}
