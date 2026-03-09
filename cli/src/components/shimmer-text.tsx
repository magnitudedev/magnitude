import { TextAttributes } from '@opentui/core'
import React, { useEffect, useMemo, useState } from 'react'

import { useTheme } from '../hooks/use-theme'
import {
  clampRange,
  convertHslToRgb,
  convertRgbToHsl,
  formatRgbToHex,
  parseHexColorToRgb,
} from '../utils/color-conversion'

const buildPaletteFromPrimaryColor = (
  primaryColor: string,
  size: number,
  fallbackColor: string,
): string[] => {
  const baseRgb = parseHexColorToRgb(primaryColor)
  if (!baseRgb) {
    // If we can't parse the color, return a simple palette using the fallback
    return Array.from({ length: size }, () => fallbackColor)
  }

  const { h, s, l } = convertRgbToHsl(baseRgb.r, baseRgb.g, baseRgb.b)
  const palette: string[] = []
  const paletteSize = Math.max(6, Math.min(24, size))
  const lightnessRange = 0.22

  for (let i = 0; i < paletteSize; i++) {
    const ratio = paletteSize === 1 ? 0.5 : i / (paletteSize - 1)
    const offset = (0.5 - ratio) * 2 * lightnessRange
    const adjustedLightness = clampRange(l + offset, 0.08, 0.92)
    // Keep full saturation for vibrant colors, only vary lightness
    const adjustedSaturation = s
    const { r, g, b } = convertHslToRgb(h, adjustedSaturation, adjustedLightness)
    palette.push(formatRgbToHex(r, g, b))
  }

  return palette
}

const buildGradientColors = (
  length: number,
  colorPalette: string[],
  mutedColor: string,
): string[] => {
  if (length === 0) return []
  if (colorPalette.length === 0) {
    return Array.from({ length }, () => mutedColor)
  }
  if (colorPalette.length === 1) {
    return Array.from({ length }, () => colorPalette[0])
  }
  const generatedColors: string[] = []
  for (let i = 0; i < length; i++) {
    const ratio = length === 1 ? 0 : i / (length - 1)
    const colorIndex = Math.min(
      colorPalette.length - 1,
      Math.floor(ratio * (colorPalette.length - 1)),
    )
    generatedColors.push(colorPalette[colorIndex])
  }
  return generatedColors
}

const buildGradientAttributes = (length: number): number[] => {
  const attributes: number[] = []
  for (let i = 0; i < length; i++) {
    const ratio = length <= 1 ? 0 : i / (length - 1)
    if (ratio < 0.23) {
      attributes.push(TextAttributes.BOLD)
    } else if (ratio < 0.69) {
      attributes.push(TextAttributes.NONE)
    } else {
      attributes.push(TextAttributes.DIM)
    }
  }
  return attributes
}

export const ShimmerText = ({
  text,
  interval = 180,
  colors,
  primaryColor,
}: {
  text: string
  interval?: number
  colors?: string[]
  primaryColor?: string
}) => {
  const theme = useTheme()
  const [animationPhase, setAnimationPhase] = useState<number>(0)
  const chars = text.split('')
  const numChars = chars.length

  useEffect(() => {
    const pulseInterval = setInterval(() => {
      setAnimationPhase((prev) => (prev + 1) % numChars)
    }, interval)

    return () => clearInterval(pulseInterval)
  }, [interval, numChars])

  const palette = useMemo(() => {
    if (colors && colors.length > 0) {
      return colors
    }
    if (primaryColor) {
      const paletteSize = Math.max(8, Math.min(20, Math.ceil(numChars * 1.5)))
      return buildPaletteFromPrimaryColor(primaryColor, paletteSize, theme.muted)
    }
    // Use theme shimmer color as default
    const paletteSize = Math.max(8, Math.min(20, Math.ceil(numChars * 1.5)))
    return buildPaletteFromPrimaryColor(theme.info, paletteSize, theme.muted)
  }, [colors, primaryColor, numChars, theme.info, theme.muted])

  const generatedColors = useMemo(
    () => buildGradientColors(numChars, palette, theme.muted),
    [numChars, palette, theme.muted],
  )
  const attributes = useMemo(() => buildGradientAttributes(numChars), [numChars])

  const segments: { text: string; color: string; attr: number }[] = []
  let currentColor = generatedColors[0]
  let currentAttr = attributes[0]
  let currentText = ''

  chars.forEach((char, index) => {
    const phase = (animationPhase - index + numChars) % numChars
    const charColor = generatedColors[phase]
    const charAttr = attributes[phase]

    if (charColor === currentColor && charAttr === currentAttr) {
      currentText += char
    } else {
      if (currentText) {
        segments.push({
          text: currentText,
          color: currentColor,
          attr: currentAttr,
        })
      }
      currentText = char
      currentColor = charColor
      currentAttr = charAttr
    }
  })

  if (currentText) {
    segments.push({ text: currentText, color: currentColor, attr: currentAttr })
  }

  return (
    <>
      {segments.map((part, index) => (
        <span key={index} fg={part.color} attributes={part.attr}>
          {part.text}
        </span>
      ))}
    </>
  )
}