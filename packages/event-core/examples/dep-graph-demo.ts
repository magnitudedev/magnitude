/**
 * Simple agent for testing dep-graph script.
 *
 * Test with:
 *   npx tsx scripts/dep-graph.ts examples/dep-graph-demo.ts DemoAgent
 */

import { Projection, Agent } from '../src'

// Events
interface AppEvent {
  type: 'update'
}

// Config projection - no dependencies
const ConfigProjection = Projection.define<AppEvent, { value: number }>()({
  name: 'Config',
  initial: { value: 0 },
  eventHandlers: {
    update: ({ state }) => ({ value: state.value + 1 })
  }
})

// Data projection - reads from Config
const DataProjection = Projection.define<AppEvent, { data: string }>()({
  name: 'Data',
  initial: { data: '' },
  reads: [ConfigProjection] as const,
  eventHandlers: {
    update: ({ state, read }) => {
      const config = read(ConfigProjection)
      return { data: `value: ${config.value}` }
    }
  }
})

// Stats projection - reads from Data (chain: Stats -> Data -> Config)
const StatsProjection = Projection.define<AppEvent, { count: number }>()({
  name: 'Stats',
  initial: { count: 0 },
  reads: [DataProjection] as const,
  eventHandlers: {
    update: ({ state, read }) => {
      const data = read(DataProjection)
      return { count: data.data.length }
    }
  }
})

// Agent with dependency chain
export const DemoAgent = Agent.define<AppEvent>()({
  name: 'DemoAgent',
  projections: [ConfigProjection, DataProjection, StatsProjection],
  workers: [],
  expose: {}
})
