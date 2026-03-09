import React, { useState, useEffect } from 'react'

const FRAME_MS = 80
const WAVE_CHARS = 3
const SUB_COLS = WAVE_CHARS * 2  // braille: 2 sub-columns per character
const SUB_ROWS = 4               // braille: 4 sub-rows per character row
const BASELINE = 1.5             // middle of 0-3 range
const AMPLITUDE = 1.0

// Braille dot bit values: [col0, col1] for each of 4 sub-rows
const BRAILLE_BASE = 0x2800
const BRAILLE_DOTS: number[][] = [
  [1, 8],     // sub-row 0 (top)
  [2, 16],    // sub-row 1
  [4, 32],    // sub-row 2
  [64, 128],  // sub-row 3 (bottom)
]

function computeWave(phase: number): string {
  const charCodes: number[] = new Array(WAVE_CHARS).fill(0)

  for (let sc = 0; sc < SUB_COLS; sc++) {
    // Two sine waves at different frequencies for organic feel
    const t = (sc / SUB_COLS) * Math.PI * 2
    const wave = Math.sin(t * 1.5 + phase) * 0.6 + Math.sin(t * 2.8 + phase * 1.3) * 0.4
    const subRow = Math.round(BASELINE + wave * AMPLITUDE)
    const clamped = Math.max(0, Math.min(SUB_ROWS - 1, subRow))

    const charIdx = Math.floor(sc / 2)
    const colInChar = sc % 2
    charCodes[charIdx] |= BRAILLE_DOTS[clamped][colInChar]
  }

  return charCodes.map(code => String.fromCharCode(BRAILLE_BASE + code)).join('')
}

interface MiniWaveProps {
  color: string
}

export const MiniWave = ({ color }: MiniWaveProps) => {
  const [phase, setPhase] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase(prev => prev + 0.3)
    }, FRAME_MS)
    return () => clearInterval(interval)
  }, [])

  const waveText = computeWave(phase)

  return <span fg={color}>{waveText}</span>
}
