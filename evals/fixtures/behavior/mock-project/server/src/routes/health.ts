import { Elysia } from 'elysia'

export const healthRoutes = new Elysia({ prefix: '/health' }).get('/', () => ({
  status: 'ok',
  service: 'task-manager-server',
}))