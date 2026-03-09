/**
 * Circle target definitions for visual grounding eval.
 *
 * Each viewport size has its own set of circle positions.
 * Circles are asymmetrically placed in 4 quadrants with varying sizes.
 */

export interface CircleTarget {
  label: string
  centerX: number
  centerY: number
  radius: number
  color: string
}

export interface ViewportConfig {
  width: number
  height: number
  targets: CircleTarget[]
}

export const VIEWPORTS: Record<string, ViewportConfig> = {
  '1024x768': {
    width: 1024,
    height: 768,
    targets: [
      { label: 'A', centerX: 200, centerY: 140, radius: 20, color: '#e74c3c' },
      { label: 'B', centerX: 715, centerY: 215, radius: 15, color: '#2ecc71' },
      { label: 'C', centerX: 325, centerY: 545, radius: 25, color: '#3498db' },
      { label: 'D', centerX: 798, centerY: 468, radius: 18, color: '#f39c12' },
    ],
  },
  '1280x720': {
    width: 1280,
    height: 720,
    targets: [
      { label: 'A', centerX: 250, centerY: 131, radius: 20, color: '#e74c3c' },
      { label: 'B', centerX: 893, centerY: 201, radius: 15, color: '#2ecc71' },
      { label: 'C', centerX: 406, centerY: 511, radius: 25, color: '#3498db' },
      { label: 'D', centerX: 998, centerY: 439, radius: 18, color: '#f39c12' },
    ],
  },
  '1920x1080': {
    width: 1920,
    height: 1080,
    targets: [
      { label: 'A', centerX: 375, centerY: 197, radius: 20, color: '#e74c3c' },
      { label: 'B', centerX: 1340, centerY: 302, radius: 15, color: '#2ecc71' },
      { label: 'C', centerX: 609, centerY: 766, radius: 25, color: '#3498db' },
      { label: 'D', centerX: 1497, centerY: 658, radius: 18, color: '#f39c12' },
    ],
  },
}

export const VIEWPORT_IDS = Object.keys(VIEWPORTS)
