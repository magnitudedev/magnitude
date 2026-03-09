import { hash } from 'bcryptjs'
import { userService } from '../services/user-service'

export async function registerHandler({ body, set }: { body: { email?: string; password?: string; name?: string }; set: { status?: number } }) {
  const { email, password, name } = body

  if (!email || !password || !name) {
    set.status = 400
    return { error: 'email, password, and name are required' }
  }

  const existing = await userService.getByEmail(email)
  if (existing) {
    set.status = 409
    return { error: 'email already in use' }
  }

  const hashed = await hash(password, 10)
  const user = await userService.create({ email, password: hashed, name })
  return { id: user.id, email: user.email, name: user.name, role: user.role }
}