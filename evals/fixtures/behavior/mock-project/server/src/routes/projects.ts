import { Elysia } from 'elysia'
import { authGuard } from '../auth/middleware'
import { projectService } from '../services/project-service'

export const projectRoutes = new Elysia({ prefix: '/api/projects' })
  .use(authGuard)
  .get('/', async ({ user }) => projectService.list(user.id))
  .post('/', async ({ body, user, set }) => {
    const payload = body as { name?: string; description?: string }
    if (!payload.name) {
      set.status = 400
      return { error: 'name is required' }
    }
    return projectService.create({ name: payload.name, description: payload.description, ownerId: user.id })
  })
  .get('/:id', async ({ params, set }) => {
    const project = await projectService.getById(params.id)
    if (!project) {
      set.status = 404
      return { error: 'project not found' }
    }
    return project
  })
  .patch('/:id', async ({ params, body, set }) => {
    const updated = await projectService.update(params.id, body as { name?: string; description?: string | null })
    if (!updated) {
      set.status = 404
      return { error: 'project not found' }
    }
    return updated
  })
  .delete('/:id', async ({ params, set }) => {
    const deleted = await projectService.delete(params.id)
    if (!deleted) {
      set.status = 404
      return { error: 'project not found' }
    }
    return deleted
  })
  .post('/:id/members', async ({ params, body, set }) => {
    const userId = (body as { userId?: string }).userId
    if (!userId) {
      set.status = 400
      return { error: 'userId is required' }
    }
    return projectService.addMember(params.id, userId)
  })
  .delete('/:id/members/:userId', async ({ params, set }) => {
    const removed = await projectService.removeMember(params.id, params.userId)
    if (!removed) {
      set.status = 404
      return { error: 'membership not found' }
    }
    return removed
  })