
/**
 * Artifact Hooks
 *
 * Context provider and hook for accessing artifact state from the agent.
 */

import { createContext, useContext } from 'react'
import type { ArtifactState } from '@magnitudedev/agent'

const ArtifactContext = createContext<ArtifactState | null>(null)

export const ArtifactProvider = ArtifactContext.Provider

export function useArtifacts(): ArtifactState | null {
  return useContext(ArtifactContext)
}
