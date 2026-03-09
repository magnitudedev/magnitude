import { describe, expect, it } from 'bun:test'
import { taskService } from '../src/services/task-service'

describe('task service', () => {
  it('exports create', () => {
    expect(typeof taskService.create).toBe('function')
  })
})