import { Elysia } from 'elysia'
import { loginHandler } from '../auth/login'
import { registerHandler } from '../auth/register'

export const authRoutes = new Elysia({ prefix: '/auth' })
  .post('/register', registerHandler)
  .post('/login', loginHandler)