import { eq } from 'drizzle-orm'
import { db } from '../db/connection'
import { users } from '../db/schema'

export const userService = {
  async getById(id: string) {
    const [user] = await db.select().from(users).where(eq(users.id, id)).limit(1)
    return user ?? null
  },

  async getByEmail(email: string) {
    const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1)
    return user ?? null
  },

  async create(input: { email: string; password: string; name: string; role?: 'admin' | 'member' }) {
    const id = crypto.randomUUID()
    const [created] = await db
      .insert(users)
      .values({
        id,
        email: input.email,
        password: input.password,
        name: input.name,
        role: input.role ?? 'member',
        createdAt: new Date(),
      })
      .returning()
    return created
  },
}