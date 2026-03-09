/**
 * Generate the circles HTML page for a given viewport config.
 */

import type { ViewportConfig } from './targets'

export function generateCirclesHtml(config: ViewportConfig): string {
  const circlesDivs = config.targets.map(t => {
    const left = t.centerX - t.radius
    const top = t.centerY - t.radius
    const size = t.radius * 2
    return `  <div class="circle" style="
    left: ${left}px; top: ${top}px;
    width: ${size}px; height: ${size}px;
    background: ${t.color};
    font-size: ${Math.max(12, t.radius - 2)}px;
  ">${t.label}</div>`
  }).join('\n')

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      width: ${config.width}px;
      height: ${config.height}px;
      background: #ffffff;
      overflow: hidden;
      position: relative;
      font-family: Arial, sans-serif;
    }
    .circle {
      position: absolute;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: bold;
      text-shadow: 0 0 3px rgba(0,0,0,0.5);
    }
  </style>
</head>
<body>
${circlesDivs}
</body>
</html>`
}
