import { and, eq } from 'drizzle-orm'
import { db } from '../db/connection'
import { tasks } from '../db/schema'

export const taskService = {
  async listByProject(projectId: string) {
    return db.select().from(tasks).where(eq(tasks.projectId, projectId))
  },

  async getById(id: string) {
    const [task] = await db.select().from(tasks).where(eq(tasks.id, id)).limit(1)
    return task ?? null
  },

  async create(input: {
    title: string
    description?: string
    projectId: string
    assigneeId?: string | null
  }) {
    const id = crypto.randomUUID()
    const [created] = await db
      .insert(tasks)
      .values({
        id,
        title: input.title,
        description: input.description ?? null,
        projectId: input.projectId,
        assigneeId: input.assigneeId ?? null,
        status: 'todo',
        createdAt: new Date(),
      })
      .returning()
    return created
  },

  async update(id: string, input: { title?: string; description?: string | null }) {
    const [updated] = await db.update(tasks).set(input).where(eq(tasks.id, id)).returning()
    return updated ?? null
  },

  async updateStatus(id: string, status: 'todo' | 'in-progress' | 'done') {
    const [updated] = await db.update(tasks).set({ status }).where(eq(tasks.id, id)).returning()
    return updated ?? null
  },

  async assign(id: string, assigneeId: string | null) {
    const [updated] = await db.update(tasks).set({ assigneeId }).where(eq(tasks.id, id)).returning()
    return updated ?? null
  },

  async delete(id: string, projectId: string) {
    const [deleted] = await db
      .delete(tasks)
      .where(and(eq(tasks.id, id), eq(tasks.projectId, projectId)))
      .returning()
    return deleted ?? null
  },
}