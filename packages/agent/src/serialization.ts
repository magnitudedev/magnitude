/**
 * Event Serialization/Deserialization Helpers
 * 
 * Handles conversion between AppEvent instances and JSON-serializable format
 * for persistence to disk.
 */

import type { AppEvent } from './events'
import type { Timestamped } from '@magnitudedev/event-core'

export interface SerializedEvent {
  readonly type: string
  readonly timestamp: number
  readonly [key: string]: unknown
}

/**
 * Serialize an event for persistence
 * Events are guaranteed to have timestamps added by the framework (EventBusCore.publish)
 */
export function serializeEvent(event: Timestamped<AppEvent>): SerializedEvent {
  return {
    ...event,
  } as SerializedEvent
}

/**
 * Serialize multiple events for persistence
 */
export function serializeEvents(events: readonly Timestamped<AppEvent>[]): SerializedEvent[] {
  return events.map(serializeEvent)
}

/**
 * Deserialize an event from persisted format
 * Validates basic structure but trusts type discriminator
 */
export function deserializeEvent(serialized: SerializedEvent): AppEvent {
  // Basic validation
  if (!serialized || typeof serialized !== 'object') {
    throw new Error('Invalid serialized event: not an object')
  }
  
  if (typeof serialized.type !== 'string') {
    throw new Error('Invalid serialized event: missing or invalid type')
  }
  
  if (typeof serialized.timestamp !== 'number') {
    throw new Error('Invalid serialized event: missing or invalid timestamp')
  }
  
  // Trust the type discriminator and return as AppEvent
  // We use unknown cast to satisfy TypeScript's strict type checking
  return serialized as unknown as AppEvent
}

/**
 * Deserialize multiple events from persisted format
 * Returns valid events and logs errors for invalid ones
 */
export function deserializeEvents(serialized: readonly SerializedEvent[]): AppEvent[] {
  const events: AppEvent[] = []
  
  for (let i = 0; i < serialized.length; i++) {
    try {
      events.push(deserializeEvent(serialized[i]))
    } catch (error) {
      console.error(`Failed to deserialize event at index ${i}:`, error)
      // Skip invalid events but continue processing
    }
  }
  
  return events
}

/**
 * Validate that events are in chronological order
 * Returns true if valid, false otherwise
 */
export function validateEventOrder(events: readonly SerializedEvent[]): boolean {
  for (let i = 1; i < events.length; i++) {
    if (events[i].timestamp < events[i - 1].timestamp) {
      console.warn(`Event order violation at index ${i}: ${events[i].timestamp} < ${events[i - 1].timestamp}`)
      return false
    }
  }
  return true
}

/**
 * Test that an event can be serialized and deserialized without loss
 * Useful for validation during development
 */
export function testEventRoundTrip(event: Timestamped<AppEvent>): boolean {
  try {
    const serialized = serializeEvent(event)
    const deserialized = deserializeEvent(serialized)
    
    // Check that all original properties are preserved
    for (const key of Object.keys(event)) {
      if (JSON.stringify((event as any)[key]) !== JSON.stringify((deserialized as any)[key])) {
        console.error(`Round-trip mismatch for key '${key}'`)
        return false
      }
    }
    
    return true
  } catch (error) {
    console.error('Round-trip test failed:', error)
    return false
  }
}

