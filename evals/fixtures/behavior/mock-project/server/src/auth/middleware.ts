import { Elysia } from 'elysia'
import { userService } from '../services/user-service'

export const authGuard = new Elysia({ name: 'auth-guard' }).derive(async (ctx) => {
  const authHeader = ctx.headers.authorization
  if (!authHeader?.startsWith('Bearer ')) {
    ctx.set.status = 401
    throw new Error('Missing bearer token')
  }

  const token = authHeader.slice(7)
  const payload = await ctx.jwt.verify(token)

  if (!payload || typeof payload !== 'object' || !('sub' in payload)) {
    ctx.set.status = 401
    throw new Error('Invalid token')
  }

  const userId = String(payload.sub)
  const user = await userService.getById(userId)
  if (!user) {
    ctx.set.status = 401
    throw new Error('User not found')
  }

  return { user }
})