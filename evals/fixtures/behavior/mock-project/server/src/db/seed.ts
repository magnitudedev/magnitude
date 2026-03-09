import { hashSync } from 'bcryptjs'
import { db } from './connection'
import { users } from './schema'

await db.insert(users).values({
  id: crypto.randomUUID(),
  email: 'admin@example.com',
  password: hashSync('password123', 10),
  name: 'Admin User',
  role: 'admin',
  createdAt: new Date(),
})

console.log('Seed complete')