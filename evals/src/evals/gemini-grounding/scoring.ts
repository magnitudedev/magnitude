/**
 * Response parsing and scoring for Gemini grounding eval.
 *
 * Reuses the scoring logic from visual-grounding but re-exports
 * for clean module boundaries. Parsing handles the same formats
 * since Gemini should follow the XML prompt format.
 */

export { parseCoordinates, scoreCircle, scoreAllCircles } from '../visual-grounding/scoring'
export type { ParsedCoordinate, CircleScore } from '../visual-grounding/scoring'
