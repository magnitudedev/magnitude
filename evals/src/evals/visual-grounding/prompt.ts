/**
 * Prompts for the visual grounding eval.
 */

export const SYSTEM_PROMPT = `You are a visual analysis assistant. You are looking at a screenshot of a web page. Your task is to identify visual elements and report their exact pixel coordinates.

When reporting coordinates, use the exact XML format specified. Coordinates should be in pixels relative to the top-left corner of the image (0,0 is top-left).`

export const USER_PROMPT = `Look at this screenshot carefully. There are four colored circles on the page, each labeled with a letter (A, B, C, D).

For each circle, identify the exact center pixel coordinates where you would click to hit the center of that circle.

Report your answer in this exact XML format:

<coordinates>
  <circle label="A" x="___" y="___" />
  <circle label="B" x="___" y="___" />
  <circle label="C" x="___" y="___" />
  <circle label="D" x="___" y="___" />
</coordinates>

Replace ___ with the integer pixel coordinates. Be as precise as possible. Do not explain your reasoning - just output the coordinates in the XML format above.`
