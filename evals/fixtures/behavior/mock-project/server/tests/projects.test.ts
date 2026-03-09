import { describe, expect, it } from 'bun:test'
import { projectService } from '../src/services/project-service'

describe('project service', () => {
  it('exports list', () => {
    expect(typeof projectService.list).toBe('function')
  })
})