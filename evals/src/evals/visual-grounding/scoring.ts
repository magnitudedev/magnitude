/**
 * Response parsing and scoring for visual grounding eval.
 */

import type { CircleTarget } from './targets'

export interface ParsedCoordinate {
  label: string
  x: number
  y: number
}

export interface CircleScore {
  label: string
  predictedX: number | null
  predictedY: number | null
  actualX: number
  actualY: number
  distance: number | null
  score: number
  withinRadius: boolean
}

/**
 * Parse coordinates from model response.
 * Tries XML format first, falls back to common text patterns.
 */
export function parseCoordinates(raw: string): ParsedCoordinate[] {
  const results: ParsedCoordinate[] = []

  // Try XML format: <circle label="A" x="200" y="140" />
  const xmlPattern = /<circle\s+label=["']([A-Da-d])["']\s+x=["'](\d+(?:\.\d+)?)["']\s+y=["'](\d+(?:\.\d+)?)["']\s*\/?>/gi
  let match: RegExpExecArray | null
  while ((match = xmlPattern.exec(raw)) !== null) {
    results.push({
      label: match[1].toUpperCase(),
      x: Math.round(parseFloat(match[2])),
      y: Math.round(parseFloat(match[3])),
    })
  }

  if (results.length > 0) return results

  // Fallback: "A: (200, 140)" or "A: 200, 140" or "A: x=200, y=140"
  const fallbackPattern = /\b([A-Da-d])\s*[:\-]\s*\(?(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)?/gi
  while ((match = fallbackPattern.exec(raw)) !== null) {
    results.push({
      label: match[1].toUpperCase(),
      x: Math.round(parseFloat(match[2])),
      y: Math.round(parseFloat(match[3])),
    })
  }

  if (results.length > 0) return results

  // Fallback: browser.click(200, 140) style — grab all click coords and assign A-D in order
  const clickPattern = /click\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)/gi
  const labels = ['A', 'B', 'C', 'D']
  let clickIdx = 0
  while ((match = clickPattern.exec(raw)) !== null && clickIdx < labels.length) {
    results.push({
      label: labels[clickIdx],
      x: Math.round(parseFloat(match[1])),
      y: Math.round(parseFloat(match[2])),
    })
    clickIdx++
  }

  return results
}

/** Max distance (beyond radius) at which score reaches 0 */
const MAX_DISTANCE = 150

/**
 * Score a single circle prediction.
 * - Within radius: 1.0
 * - Beyond radius: linear decay to 0.0 over MAX_DISTANCE px
 * - Missing: 0.0
 */
export function scoreCircle(
  predicted: ParsedCoordinate | undefined,
  target: CircleTarget,
): CircleScore {
  if (!predicted) {
    return {
      label: target.label,
      predictedX: null,
      predictedY: null,
      actualX: target.centerX,
      actualY: target.centerY,
      distance: null,
      score: 0,
      withinRadius: false,
    }
  }

  const dx = predicted.x - target.centerX
  const dy = predicted.y - target.centerY
  const distance = Math.sqrt(dx * dx + dy * dy)
  const withinRadius = distance <= target.radius

  let score: number
  if (withinRadius) {
    score = 1.0
  } else {
    const overshoot = distance - target.radius
    score = Math.max(0, 1.0 - overshoot / MAX_DISTANCE)
  }

  return {
    label: target.label,
    predictedX: predicted.x,
    predictedY: predicted.y,
    actualX: target.centerX,
    actualY: target.centerY,
    distance: Math.round(distance * 10) / 10,
    score: Math.round(score * 1000) / 1000,
    withinRadius,
  }
}

/**
 * Score all circles for a given viewport's targets.
 */
export function scoreAllCircles(
  parsed: ParsedCoordinate[],
  targets: CircleTarget[],
): CircleScore[] {
  const byLabel = new Map(parsed.map(p => [p.label, p]))
  return targets.map(target => scoreCircle(byLabel.get(target.label), target))
}
