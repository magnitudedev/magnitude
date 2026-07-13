/**
 * Shared Core Types
 */

// Re-export BaseEvent from event-bus-core as the canonical event constraint
export type { BaseEvent } from './core/event-bus-core'

// Core Flow Types
export type FlowStatus = 'working' | 'pending_user_action' | 'resolved' | 'failed'

export interface Binding {
  type: string
  id: string
}