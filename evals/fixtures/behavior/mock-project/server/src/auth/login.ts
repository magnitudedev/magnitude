import { compare } from 'bcryptjs'
import { userService } from '../services/user-service'

export async function loginHandler({
  body,
  jwt,
  set,
}: {
  body: { email?: string; password?: string }
  jwt: { sign: (value: Record<string, unknown>) => Promise<string> }
  set: { status?: number }
}) {
  const { email, password } = body
  if (!email || !password) {
    set.status = 400
    return { error: 'email and password are required' }
  }

  const user = await userService.getByEmail(email)
  if (!user || !(await compare(password, user.password))) {
    set.status = 401
    return { error: 'invalid credentials' }
  }

  const token = await jwt.sign({ sub: user.id, email: user.email, role: user.role })
  return { token, user: { id: user.id, email: user.email, name: user.name, role: user.role } }
}