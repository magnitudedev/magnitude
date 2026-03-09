import { describe, expect, it } from 'bun:test'
import { app } from '../src/index'

describe('auth routes', () => {
  it('has health route', async () => {
    const res = await app.handle(new Request('http://localhost/health'))
    expect(res.status).toBe(200)
  })
})