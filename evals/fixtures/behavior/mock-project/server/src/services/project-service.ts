import { and, eq, inArray } from 'drizzle-orm'
import { db } from '../db/connection'
import { projectMembers, projects } from '../db/schema'

export const projectService = {
  async list(userId: string) {
    const memberships = await db
      .select({ projectId: projectMembers.projectId })
      .from(projectMembers)
      .where(eq(projectMembers.userId, userId))

    const memberProjectIds = memberships.map((m) => m.projectId)

    if (memberProjectIds.length === 0) {
      return db.select().from(projects).where(eq(projects.ownerId, userId))
    }

    return db
      .select()
      .from(projects)
      .where(
        inArray(
          projects.id,
          Array.from(new Set([...memberProjectIds])),
        ),
      )
  },

  async getById(id: string) {
    const [project] = await db.select().from(projects).where(eq(projects.id, id)).limit(1)
    return project ?? null
  },

  async create(input: { name: string; description?: string; ownerId: string }) {
    const id = crypto.randomUUID()
    const [created] = await db
      .insert(projects)
      .values({
        id,
        name: input.name,
        description: input.description ?? null,
        ownerId: input.ownerId,
        createdAt: new Date(),
      })
      .returning()

    await db.insert(projectMembers).values({
      projectId: id,
      userId: input.ownerId,
    })

    return created
  },

  async update(id: string, input: { name?: string; description?: string | null }) {
    const [updated] = await db
      .update(projects)
      .set(input)
      .where(eq(projects.id, id))
      .returning()
    return updated ?? null
  },

  async delete(id: string) {
    await db.delete(projectMembers).where(eq(projectMembers.projectId, id))
    const [deleted] = await db.delete(projects).where(eq(projects.id, id)).returning()
    return deleted ?? null
  },

  async addMember(projectId: string, userId: string) {
    const [existing] = await db
      .select()
      .from(projectMembers)
      .where(and(eq(projectMembers.projectId, projectId), eq(projectMembers.userId, userId)))
      .limit(1)

    if (!existing) {
      await db.insert(projectMembers).values({ projectId, userId })
    }
    return { projectId, userId }
  },

  async removeMember(projectId: string, userId: string) {
    const [removed] = await db
      .delete(projectMembers)
      .where(and(eq(projectMembers.projectId, projectId), eq(projectMembers.userId, userId)))
      .returning()
    return removed ?? null
  },
}