import { Elysia } from 'elysia'
import { authGuard } from '../auth/middleware'
import { taskService } from '../services/task-service'

export const taskRoutes = new Elysia({ prefix: '/api/projects/:projectId/tasks' })
  .use(authGuard)
  .get('/', async ({ params }) => taskService.listByProject(params.projectId))
  .post('/', async ({ params, body, set }) => {
    const payload = body as { title?: string; description?: string; assigneeId?: string | null }
    if (!payload.title) {
      set.status = 400
      return { error: 'title is required' }
    }
    return taskService.create({
      title: payload.title,
      description: payload.description,
      assigneeId: payload.assigneeId ?? null,
      projectId: params.projectId,
    })
  })
  .patch('/:taskId', async ({ params, body, set }) => {
    const updated = await taskService.update(params.taskId, body as { title?: string; description?: string | null })
    if (!updated) {
      set.status = 404
      return { error: 'task not found' }
    }
    return updated
  })
  .patch('/:taskId/status', async ({ params, body, set }) => {
    const status = (body as { status?: 'todo' | 'in-progress' | 'done' }).status
    if (!status) {
      set.status = 400
      return { error: 'status is required' }
    }
    const updated = await taskService.updateStatus(params.taskId, status)
    if (!updated) {
      set.status = 404
      return { error: 'task not found' }
    }
    return updated
  })
  .patch('/:taskId/assign', async ({ params, body, set }) => {
    const assigneeId = (body as { assigneeId?: string | null }).assigneeId ?? null
    const updated = await taskService.assign(params.taskId, assigneeId)
    if (!updated) {
      set.status = 404
      return { error: 'task not found' }
    }
    return updated
  })
  .delete('/:taskId', async ({ params, set }) => {
    const deleted = await taskService.delete(params.taskId, params.projectId)
    if (!deleted) {
      set.status = 404
      return { error: 'task not found' }
    }
    return deleted
  })