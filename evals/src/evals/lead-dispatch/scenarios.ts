/**
 * Orchestrator Dispatch Eval Scenarios
 *
 * 12 scenarios across 6 task categories + lifecycle tests.
 * Multi-turn scenarios provide syntheticResponses so the turn loop
 * can feed realistic agent output back to the lead.
 *
 * Agent responses use the artifact pattern: agents write detailed reports
 * into artifacts and send a brief message back. The lead receives
 * the message + artifact content as separate XML blocks.
 */

import type { Scenario } from '../../types'
import {
  hasLensesBlock,
  hasUserMessage,
  hasMessageToUser,
  noAgentsDeployed,
  agentTypeDeployed,
  agentTypeNotDeployed,
  agentOrderedBefore,
  reviewerDeployedAfterBuilder,
  usesReadOnlyInvestigationBeforeExecution,
  hasNoDirectMutationToolsBeforeApproval,
  usedDirectTools,
} from './checks'

// =============================================================================
// Dispatch scenario type
// =============================================================================

export interface SyntheticAgentResponse {
  /** Brief completion message (what goes inside <agent_response>) */
  message: string
  /** Content to write into the agent's writable artifact. The artifact ID
   *  comes from whatever the lead passes as <writable-artifact>. */
  artifactContent?: string
}

export interface DispatchScenario extends Scenario {
  /** Agent type → synthetic response. Fed back when the lead deploys that agent type. */
  syntheticResponses?: Record<string, SyntheticAgentResponse>
  /** Optional scripted approval/rejection user message injected by runtime. */
  conversationScript?: { approval?: string; rejection?: string; injectAfter?: number | 'first-plan-message' }
  /** Optional lifecycle completion requirements. */
  completionExpectations?: { requireBuilder?: boolean; requireReviewer?: boolean }
  /** Mock file contents for the fake project. Path → content. Used when the lead reads files directly. */
  mockFiles?: Record<string, string>
}

// =============================================================================
// Shared session context
// =============================================================================

const SESSION_CONTEXT = `<session_context>
Full name: Alex Chen
Timezone: America/Los_Angeles
Working directory: /Users/alex/myapp
Shell: zsh
Username: alex
Platform: macos
Git branch: main
Git status:
(clean)

Recent commits:
a1b2c3d Add user authentication
e4f5g6h Set up database models
d3e4f5g Add post and comment routes
b2c3d4e Add middleware and services layer
i7j8k9l Initial project scaffold

Folder structure:
src/
  index.ts
  app.ts
  auth/
    login.ts
    register.ts
    middleware.ts
    password-reset.ts
    types.ts
    index.ts
  models/
    user.ts
    post.ts
    comment.ts
    index.ts
  routes/
    api.ts
    auth.ts
    health.ts
    posts.ts
    comments.ts
    admin.ts
  middleware/
    error-handler.ts
    request-logger.ts
    cors.ts
    validate.ts
    index.ts
  services/
    email.ts
    user-service.ts
    post-service.ts
    notification.ts
    cache.ts
  utils/
    logger.ts
    config.ts
    errors.ts
    pagination.ts
    crypto.ts
  db/
    connection.ts
    seed.ts
    migrations/
      001-users.sql
      002-posts.sql
      003-comments.sql
      004-notifications.sql
  types/
    express.d.ts
    api.ts
    index.ts
tests/
  setup.ts
  auth.test.ts
  routes.test.ts
  services.test.ts
prisma/
  schema.prisma
package.json
tsconfig.json
jest.config.ts
.env
.env.example
docker-compose.yml
</session_context>`

function userMsg(text: string): string {
  return `<user mode="text" at="2026-Mar-04 10:00:00">\n${text}\n</user>`
}

// =============================================================================
// Mock file contents for the fake project
// =============================================================================

const MOCK_FILES: Record<string, string> = {
  'package.json': `{
  "name": "myapp",
  "version": "1.0.0",
  "scripts": {
    "dev": "ts-node src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "test": "jest --config jest.config.ts",
    "seed": "ts-node src/db/seed.ts",
    "migrate": "prisma migrate dev"
  },
  "dependencies": {
    "express": "^4.18.2",
    "bcrypt": "^5.1.1",
    "jsonwebtoken": "^9.0.2",
    "ioredis": "^5.3.2",
    "@prisma/client": "^5.10.2",
    "cors": "^2.8.5",
    "helmet": "^7.1.0",
    "nodemailer": "^6.9.8",
    "winston": "^3.11.0",
    "uuid": "^9.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3.3",
    "@types/express": "^4.17.21",
    "@types/bcrypt": "^5.0.2",
    "@types/jsonwebtoken": "^9.0.5",
    "@types/cors": "^2.8.17",
    "@types/nodemailer": "^6.4.14",
    "@types/uuid": "^9.0.7",
    "prisma": "^5.10.2",
    "ts-node": "^10.9.2",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.11",
    "ts-jest": "^29.1.1"
  }
}`,

  'src/index.ts': `import { createApp } from './app'
import { config } from './utils/config'
import { logger } from './utils/logger'
import { prisma } from './db/connection'

const app = createApp()

app.listen(config.port, () => {
  logger.info(\`Server running on port \${config.port}\`)
})

process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully')
  await prisma.$disconnect()
  process.exit(0)
})`,

  'src/app.ts': `import express from 'express'
import { apiRouter } from './routes/api'
import { authRouter } from './routes/auth'
import { healthRouter } from './routes/health'
import { postsRouter } from './routes/posts'
import { commentsRouter } from './routes/comments'
import { adminRouter } from './routes/admin'
import { corsMiddleware } from './middleware/cors'
import { requestLogger } from './middleware/request-logger'
import { errorHandler } from './middleware/error-handler'

export function createApp() {
  const app = express()

  // Global middleware
  app.use(express.json())
  app.use(corsMiddleware)
  app.use(requestLogger)

  // Routes
  app.use('/health', healthRouter)
  app.use('/auth', authRouter)
  app.use('/api', apiRouter)
  app.use('/api/posts', postsRouter)
  app.use('/api/posts', commentsRouter)
  app.use('/admin', adminRouter)

  // Error handling (must be last)
  app.use(errorHandler)

  return app
}`,

  'src/auth/login.ts': `import { Router } from 'express'
import bcrypt from 'bcrypt'
import jwt from 'jsonwebtoken'
import { findByEmail } from '../models/user'
import { config } from '../utils/config'
import { logger } from '../utils/logger'

export const loginRouter = Router()

loginRouter.post('/login', async (req, res) => {
  const { email, password } = req.body
  if (!email || !password) return res.status(400).json({ error: 'Missing credentials' })

  const user = await findByEmail(email)
  if (!user) return res.status(401).json({ error: 'Invalid credentials' })

  const valid = await bcrypt.compare(password, user.passwordHash)
  if (!valid) return res.status(401).json({ error: 'Invalid credentials' })

  logger.info(\`User \${user.id} logged in\`)
  const token = jwt.sign({ userId: user.id, role: user.role }, config.jwtSecret, { expiresIn: '24h' })
  res.cookie('session', token, { httpOnly: true, secure: true, sameSite: 'strict' })
  res.json({ token })
})`,

  'src/auth/register.ts': `import { Router } from 'express'
import { createUser, findByEmail } from '../models/user'
import { sendWelcomeEmail } from '../services/email'
import { logger } from '../utils/logger'

export const registerRouter = Router()

registerRouter.post('/register', async (req, res) => {
  const { email, password, name } = req.body
  if (!email || !password || !name) {
    return res.status(400).json({ error: 'Missing required fields' })
  }

  const existing = await findByEmail(email)
  if (existing) {
    return res.status(409).json({ error: 'Email already registered' })
  }

  const user = await createUser(email, password, name)
  logger.info(\`New user registered: \${user.id}\`)

  await sendWelcomeEmail(email, name)
  res.status(201).json({ id: user.id, email: user.email })
})`,

  'src/auth/middleware.ts': `import { Request, Response, NextFunction } from 'express'
import jwt from 'jsonwebtoken'
import { config } from '../utils/config'

export interface AuthenticatedRequest extends Request {
  user?: { userId: string; role: string }
}

export function authenticateToken(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  const authHeader = req.headers['authorization']
  const token = authHeader?.split(' ')[1]

  if (!token) return res.sendStatus(401)

  try {
    const decoded = jwt.verify(token, config.jwtSecret) as { userId: string; role: string }
    req.user = decoded
    next()
  } catch {
    res.sendStatus(401)
  }
}

export function requireRole(role: string) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.user || req.user.role !== role) {
      return res.status(403).json({ error: 'Insufficient permissions' })
    }
    next()
  }
}

export function validateToken(token: string): { userId: string; role: string } | null {
  try {
    return jwt.verify(token, config.jwtSecret) as { userId: string; role: string }
  } catch {
    return null
  }
}`,

  'src/auth/password-reset.ts': `import { Router } from 'express'
import { findByEmail, updatePassword } from '../models/user'
import { generateResetToken, verifyResetToken } from '../utils/crypto'
import { sendPasswordResetEmail } from '../services/email'
import { logger } from '../utils/logger'

export const passwordResetRouter = Router()

passwordResetRouter.post('/forgot-password', async (req, res) => {
  const { email } = req.body
  if (!email) return res.status(400).json({ error: 'Email required' })

  const user = await findByEmail(email)
  if (!user) return res.json({ message: 'If the email exists, a reset link was sent' })

  const token = generateResetToken(user.id)
  await sendPasswordResetEmail(email, token)
  logger.info(\`Password reset requested for \${user.id}\`)
  res.json({ message: 'If the email exists, a reset link was sent' })
})

passwordResetRouter.post('/reset-password', async (req, res) => {
  const { token, newPassword } = req.body
  if (!token || !newPassword) return res.status(400).json({ error: 'Token and password required' })

  const userId = verifyResetToken(token)
  if (!userId) return res.status(400).json({ error: 'Invalid or expired token' })

  await updatePassword(userId, newPassword)
  logger.info(\`Password reset completed for \${userId}\`)
  res.json({ message: 'Password updated' })
})`,

  'src/auth/types.ts': `export interface User {
  id: string
  email: string
  name: string
  passwordHash: string
  role: 'user' | 'admin'
  createdAt: Date
  updatedAt: Date
}

export interface TokenPayload {
  userId: string
  role: string
  iat: number
  exp: number
}`,

  'src/auth/index.ts': `export { loginRouter } from './login'
export { registerRouter } from './register'
export { passwordResetRouter } from './password-reset'
export { authenticateToken, requireRole } from './middleware'
export type { AuthenticatedRequest } from './middleware'`,

  'src/models/user.ts': `import { prisma } from '../db/connection'
import bcrypt from 'bcrypt'

export async function findByEmail(email: string) {
  return prisma.user.findUnique({ where: { email } })
}

export async function findById(id: string) {
  return prisma.user.findUnique({ where: { id } })
}

export async function createUser(email: string, password: string, name: string) {
  const passwordHash = await bcrypt.hash(password, 10)
  return prisma.user.create({ data: { email, passwordHash, name } })
}

export async function updatePassword(userId: string, newPassword: string) {
  const passwordHash = await bcrypt.hash(newPassword, 10)
  return prisma.user.update({ where: { id: userId }, data: { passwordHash } })
}

export async function getUsers(page: number = 1, limit: number = 20) {
  return prisma.user.findMany({
    skip: (page - 1) * limit,
    take: limit,
    select: { id: true, email: true, name: true, role: true, createdAt: true },
    orderBy: { createdAt: 'desc' },
  })
}

export async function getUserCount() {
  return prisma.user.count()
}`,

  'src/models/post.ts': `import { prisma } from '../db/connection'

export async function createPost(authorId: string, title: string, body: string) {
  return prisma.post.create({ data: { authorId, title, body } })
}

export async function getPostById(id: string) {
  return prisma.post.findUnique({
    where: { id },
    include: { author: { select: { id: true, name: true } } },
  })
}

export async function getPosts(page: number = 1, limit: number = 20) {
  return prisma.post.findMany({
    skip: (page - 1) * limit,
    take: limit,
    include: { author: { select: { id: true, name: true } } },
    orderBy: { createdAt: 'desc' },
  })
}

export async function updatePost(id: string, authorId: string, data: { title?: string; body?: string }) {
  return prisma.post.updateMany({
    where: { id, authorId },
    data,
  })
}

export async function deletePost(id: string, authorId: string) {
  return prisma.post.deleteMany({ where: { id, authorId } })
}

export async function getPostCount() {
  return prisma.post.count()
}`,

  'src/models/comment.ts': `import { prisma } from '../db/connection'

export async function createComment(postId: string, authorId: string, content: string) {
  return prisma.comment.create({
    data: { postId, authorId, content },
    include: { author: { select: { id: true, name: true } } },
  })
}

export async function getCommentsByPost(postId: string, page: number = 1, limit: number = 50) {
  return prisma.comment.findMany({
    where: { postId },
    skip: (page - 1) * limit,
    take: limit,
    include: { author: { select: { id: true, name: true } } },
    orderBy: { createdAt: 'asc' },
  })
}

export async function deleteComment(id: string, authorId: string) {
  return prisma.comment.deleteMany({ where: { id, authorId } })
}`,

  'src/models/index.ts': `export * from './user'
export * from './post'
export * from './comment'`,

  'src/routes/api.ts': `import { Router } from 'express'
import { authenticateToken } from '../auth/middleware'
import { UserService } from '../services/user-service'
import { paginate } from '../utils/pagination'

export const apiRouter = Router()

apiRouter.use(authenticateToken)

apiRouter.get('/users', async (req, res) => {
  const { page, limit } = paginate(req.query)
  const users = await UserService.list(page, limit)
  res.json(users)
})

apiRouter.post('/users', async (req, res) => {
  const { email, password, name } = req.body
  const user = await UserService.create(email, password, name)
  res.status(201).json(user)
})

apiRouter.get('/profile', async (req, res) => {
  res.json(req.user)
})

apiRouter.get('/users/:id', async (req, res) => {
  const user = await UserService.getById(req.params.id)
  if (!user) return res.status(404).json({ error: 'User not found' })
  res.json(user)
})`,

  'src/routes/auth.ts': `import { Router } from 'express'
import { loginRouter } from '../auth/login'
import { registerRouter } from '../auth/register'
import { passwordResetRouter } from '../auth/password-reset'

export const authRouter = Router()

authRouter.use('/', loginRouter)
authRouter.use('/', registerRouter)
authRouter.use('/', passwordResetRouter)`,

  'src/routes/health.ts': `import { Router } from 'express'

export const healthRouter = Router()

healthRouter.get('/', (req, res) => {
  res.sendStatus(200)
})`,

  'src/routes/posts.ts': `import { Router } from 'express'
import { authenticateToken, AuthenticatedRequest } from '../auth/middleware'
import { PostService } from '../services/post-service'
import { paginate } from '../utils/pagination'

export const postsRouter = Router()

postsRouter.get('/', async (req, res) => {
  const { page, limit } = paginate(req.query)
  const posts = await PostService.list(page, limit)
  res.json(posts)
})

postsRouter.get('/:id', async (req, res) => {
  const post = await PostService.getById(req.params.id)
  if (!post) return res.status(404).json({ error: 'Post not found' })
  res.json(post)
})

postsRouter.post('/', authenticateToken, async (req: AuthenticatedRequest, res) => {
  const { title, body } = req.body
  const post = await PostService.create(req.user!.userId, title, body)
  res.status(201).json(post)
})

postsRouter.put('/:id', authenticateToken, async (req: AuthenticatedRequest, res) => {
  const { title, body } = req.body
  await PostService.update(req.params.id, req.user!.userId, { title, body })
  res.json({ message: 'Updated' })
})

postsRouter.delete('/:id', authenticateToken, async (req: AuthenticatedRequest, res) => {
  await PostService.delete(req.params.id, req.user!.userId)
  res.status(204).send()
})`,

  'src/routes/comments.ts': `import { Router } from 'express'
import { authenticateToken, AuthenticatedRequest } from '../auth/middleware'
import { createComment, getCommentsByPost, deleteComment } from '../models/comment'
import { paginate } from '../utils/pagination'

export const commentsRouter = Router()

commentsRouter.get('/:postId/comments', async (req, res) => {
  const { page, limit } = paginate(req.query)
  const comments = await getCommentsByPost(req.params.postId, page, limit)
  res.json(comments)
})

commentsRouter.post('/:postId/comments', authenticateToken, async (req: AuthenticatedRequest, res) => {
  const { content } = req.body
  const comment = await createComment(req.params.postId, req.user!.userId, content)
  res.status(201).json(comment)
})

commentsRouter.delete('/:postId/comments/:commentId', authenticateToken, async (req: AuthenticatedRequest, res) => {
  await deleteComment(req.params.commentId, req.user!.userId)
  res.status(204).send()
})`,

  'src/routes/admin.ts': `import { Router } from 'express'
import { authenticateToken, requireRole } from '../auth/middleware'
import { getUsers, getUserCount } from '../models/user'
import { getPostCount } from '../models/post'
import { logger } from '../utils/logger'

export const adminRouter = Router()

adminRouter.use(authenticateToken)
adminRouter.use(requireRole('admin'))

adminRouter.get('/stats', async (req, res) => {
  const [userCount, postCount] = await Promise.all([getUserCount(), getPostCount()])
  res.json({ users: userCount, posts: postCount })
})

adminRouter.get('/users', async (req, res) => {
  const users = await getUsers(1, 100)
  res.json(users)
})

adminRouter.post('/users/:id/role', async (req, res) => {
  const { role } = req.body
  logger.info(\`Admin changing role of user \${req.params.id} to \${role}\`)
  res.json({ message: 'Role updated' })
})`,

  'src/middleware/error-handler.ts': `import { Request, Response, NextFunction } from 'express'
import { logger } from '../utils/logger'
import { AppError } from '../utils/errors'

export function errorHandler(err: Error, req: Request, res: Response, _next: NextFunction) {
  if (err instanceof AppError) {
    logger.warn(\`AppError: \${err.message}\`, { statusCode: err.statusCode, path: req.path })
    return res.status(err.statusCode).json({ error: err.message })
  }

  logger.error(\`Unhandled error: \${err.message}\`, { stack: err.stack, path: req.path })
  res.status(500).json({ error: 'Internal server error' })
}`,

  'src/middleware/request-logger.ts': `import { Request, Response, NextFunction } from 'express'
import { logger } from '../utils/logger'

export function requestLogger(req: Request, res: Response, next: NextFunction) {
  const start = Date.now()

  res.on('finish', () => {
    const duration = Date.now() - start
    logger.info(\`\${req.method} \${req.path} \${res.statusCode} \${duration}ms\`)
  })

  next()
}`,

  'src/middleware/cors.ts': `import cors from 'cors'
import { config } from '../utils/config'

export const corsMiddleware = cors({
  origin: config.corsOrigins,
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization'],
})`,

  'src/middleware/validate.ts': `import { Request, Response, NextFunction } from 'express'

// Generic request validation middleware
// Currently unused — validation is done inline in route handlers
export function validateBody(validator: (body: unknown) => { valid: boolean; errors?: string[] }) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = validator(req.body)
    if (!result.valid) {
      return res.status(400).json({ errors: result.errors })
    }
    next()
  }
}`,

  'src/middleware/index.ts': `export { errorHandler } from './error-handler'
export { requestLogger } from './request-logger'
export { corsMiddleware } from './cors'
export { validateBody } from './validate'`,

  'src/services/email.ts': `import nodemailer from 'nodemailer'
import { config } from '../utils/config'
import { logger } from '../utils/logger'

const transporter = nodemailer.createTransport({
  host: config.smtpHost,
  port: config.smtpPort,
  auth: { user: config.smtpUser, pass: config.smtpPass },
})

export async function sendWelcomeEmail(email: string, name: string) {
  await transporter.sendMail({
    from: config.emailFrom,
    to: email,
    subject: 'Welcome to MyApp',
    html: \`<h1>Welcome, \${name}!</h1><p>Your account has been created.</p>\`,
  })
  logger.info(\`Welcome email sent to \${email}\`)
}

export async function sendPasswordResetEmail(email: string, token: string) {
  const resetUrl = \`\${config.appUrl}/reset-password?token=\${token}\`
  await transporter.sendMail({
    from: config.emailFrom,
    to: email,
    subject: 'Password Reset',
    html: \`<p>Click <a href="\${resetUrl}">here</a> to reset your password. Link expires in 1 hour.</p>\`,
  })
  logger.info(\`Password reset email sent to \${email}\`)
}`,

  'src/services/user-service.ts': `import { findByEmail, findById, createUser, getUsers, getUserCount } from '../models/user'
import { AppError } from '../utils/errors'
import { logger } from '../utils/logger'

export class UserService {
  static async create(email: string, password: string, name: string) {
    const existing = await findByEmail(email)
    if (existing) throw new AppError('Email already registered', 409)

    const user = await createUser(email, password, name)
    logger.info(\`User created: \${user.id}\`)
    return { id: user.id, email: user.email, name: user.name }
  }

  static async getById(id: string) {
    const user = await findById(id)
    if (!user) return null
    return { id: user.id, email: user.email, name: user.name, role: user.role }
  }

  static async list(page: number, limit: number) {
    const [users, total] = await Promise.all([getUsers(page, limit), getUserCount()])
    return { data: users, total, page, limit }
  }
}`,

  'src/services/post-service.ts': `import { createPost, getPostById, getPosts, updatePost, deletePost, getPostCount } from '../models/post'
import { AppError } from '../utils/errors'
import { sendNotification } from './notification'

export class PostService {
  static async create(authorId: string, title: string, body: string) {
    const post = await createPost(authorId, title, body)
    await sendNotification('post:created', { postId: post.id, authorId })
    return post
  }

  static async getById(id: string) {
    return getPostById(id)
  }

  static async list(page: number, limit: number) {
    const [posts, total] = await Promise.all([getPosts(page, limit), getPostCount()])
    return { data: posts, total, page, limit }
  }

  static async update(id: string, authorId: string, data: { title?: string; body?: string }) {
    const result = await updatePost(id, authorId, data)
    if (result.count === 0) throw new AppError('Post not found or not owned by user', 404)
  }

  static async delete(id: string, authorId: string) {
    const result = await deletePost(id, authorId)
    if (result.count === 0) throw new AppError('Post not found or not owned by user', 404)
  }
}`,

  'src/services/notification.ts': `import { redis } from '../db/connection'
import { logger } from '../utils/logger'

export async function sendNotification(channel: string, payload: Record<string, unknown>) {
  try {
    await redis.publish(channel, JSON.stringify(payload))
    logger.debug(\`Notification sent: \${channel}\`)
  } catch (err) {
    logger.error(\`Failed to send notification: \${err}\`)
  }
}

export async function subscribeToNotifications(
  channel: string,
  handler: (payload: Record<string, unknown>) => void
) {
  const subscriber = redis.duplicate()
  await subscriber.subscribe(channel)
  subscriber.on('message', (_ch, message) => {
    try {
      handler(JSON.parse(message))
    } catch (err) {
      logger.error(\`Failed to process notification: \${err}\`)
    }
  })
  return subscriber
}`,

  'src/services/cache.ts': `import { redis } from '../db/connection'
import { logger } from '../utils/logger'

const DEFAULT_TTL = 300 // 5 minutes

export async function cacheGet<T>(key: string): Promise<T | null> {
  const cached = await redis.get(key)
  if (!cached) return null
  return JSON.parse(cached) as T
}

export async function cacheSet(key: string, value: unknown, ttl: number = DEFAULT_TTL): Promise<void> {
  await redis.setex(key, ttl, JSON.stringify(value))
}

export async function cacheInvalidate(pattern: string): Promise<void> {
  const keys = await redis.keys(pattern)
  if (keys.length > 0) {
    await redis.del(...keys)
    logger.debug(\`Invalidated \${keys.length} cache keys matching \${pattern}\`)
  }
}`,

  'src/utils/config.ts': `export const config = {
  port: parseInt(process.env.PORT || '3000'),
  jwtSecret: process.env.JWT_SECRET || 'dev-secret',
  databaseUrl: process.env.DATABASE_URL || 'postgresql://localhost:5432/myapp',
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
  corsOrigins: (process.env.CORS_ORIGINS || 'http://localhost:3000').split(','),
  appUrl: process.env.APP_URL || 'http://localhost:3000',
  emailFrom: process.env.EMAIL_FROM || 'noreply@myapp.com',
  smtpHost: process.env.SMTP_HOST || 'localhost',
  smtpPort: parseInt(process.env.SMTP_PORT || '587'),
  smtpUser: process.env.SMTP_USER || '',
  smtpPass: process.env.SMTP_PASS || '',
}`,

  'src/utils/logger.ts': `import winston from 'winston'

export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(winston.format.colorize(), winston.format.simple()),
    }),
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/app.log' }),
  ],
})`,

  'src/utils/errors.ts': `export class AppError extends Error {
  constructor(
    message: string,
    public statusCode: number = 500,
  ) {
    super(message)
    this.name = 'AppError'
  }
}

export class NotFoundError extends AppError {
  constructor(resource: string) {
    super(\`\${resource} not found\`, 404)
    this.name = 'NotFoundError'
  }
}

export class ValidationError extends AppError {
  constructor(
    message: string,
    public fields?: Record<string, string>,
  ) {
    super(message, 400)
    this.name = 'ValidationError'
  }
}`,

  'src/utils/pagination.ts': `export function paginate(query: Record<string, unknown>): { page: number; limit: number } {
  const page = Math.max(1, parseInt(String(query.page || '1'), 10))
  const limit = Math.min(100, Math.max(1, parseInt(String(query.limit || '20'), 10)))
  return { page, limit }
}

export function paginateResponse<T>(data: T[], total: number, page: number, limit: number) {
  return {
    data,
    meta: {
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit),
      hasMore: page * limit < total,
    },
  }
}`,

  'src/utils/crypto.ts': `import crypto from 'crypto'
import jwt from 'jsonwebtoken'
import { config } from './config'

export function generateResetToken(userId: string): string {
  return jwt.sign({ userId, purpose: 'password-reset' }, config.jwtSecret, { expiresIn: '1h' })
}

export function verifyResetToken(token: string): string | null {
  try {
    const decoded = jwt.verify(token, config.jwtSecret) as { userId: string; purpose: string }
    if (decoded.purpose !== 'password-reset') return null
    return decoded.userId
  } catch {
    return null
  }
}

export function generateId(): string {
  return crypto.randomUUID()
}`,

  'src/db/connection.ts': `import { PrismaClient } from '@prisma/client'
import Redis from 'ioredis'
import { config } from '../utils/config'

export const prisma = new PrismaClient({
  datasources: { db: { url: config.databaseUrl } },
})

export const redis = new Redis(config.redisUrl)`,

  'src/db/seed.ts': `import { prisma } from './connection'
import bcrypt from 'bcrypt'

async function seed() {
  const passwordHash = await bcrypt.hash('password123', 10)

  const admin = await prisma.user.upsert({
    where: { email: 'admin@myapp.com' },
    update: {},
    create: { email: 'admin@myapp.com', name: 'Admin', passwordHash, role: 'admin' },
  })

  const user = await prisma.user.upsert({
    where: { email: 'user@myapp.com' },
    update: {},
    create: { email: 'user@myapp.com', name: 'Test User', passwordHash, role: 'user' },
  })

  await prisma.post.create({
    data: { authorId: user.id, title: 'First Post', body: 'Hello world!' },
  })

  console.log(\`Seeded: admin=\${admin.id}, user=\${user.id}\`)
}

seed().catch(console.error).finally(() => prisma.$disconnect())`,

  'src/types/express.d.ts': `import { TokenPayload } from '../auth/types'

declare global {
  namespace Express {
    interface Request {
      user?: TokenPayload
    }
  }
}

export {}`,

  'src/types/api.ts': `export interface PaginatedResponse<T> {
  data: T[]
  meta: {
    total: number
    page: number
    limit: number
    totalPages: number
    hasMore: boolean
  }
}

export interface ErrorResponse {
  error: string
  details?: Record<string, string>
}

export interface SuccessResponse {
  message: string
}`,

  'src/types/index.ts': `export type { PaginatedResponse, ErrorResponse, SuccessResponse } from './api'`,

  'prisma/schema.prisma': `generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id           String    @id @default(uuid())
  email        String    @unique
  name         String
  passwordHash String
  role         String    @default("user")
  posts        Post[]
  comments     Comment[]
  createdAt    DateTime  @default(now())
  updatedAt    DateTime  @updatedAt
}

model Post {
  id        String    @id @default(uuid())
  title     String
  body      String
  author    User      @relation(fields: [authorId], references: [id])
  authorId  String
  comments  Comment[]
  createdAt DateTime  @default(now())
  updatedAt DateTime  @updatedAt
}

model Comment {
  id        String   @id @default(uuid())
  content   String
  author    User     @relation(fields: [authorId], references: [id])
  authorId  String
  post      Post     @relation(fields: [postId], references: [id])
  postId    String
  createdAt DateTime @default(now())
}`,

  'jest.config.ts': `export default {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  setupFilesAfterSetup: ['<rootDir>/tests/setup.ts'],
  coverageDirectory: 'coverage',
  collectCoverageFrom: ['src/**/*.ts', '!src/index.ts'],
}`,

  'tests/setup.ts': `import { prisma } from '../src/db/connection'

beforeEach(async () => {
  await prisma.comment.deleteMany()
  await prisma.post.deleteMany()
  await prisma.user.deleteMany()
})

afterAll(async () => {
  await prisma.$disconnect()
})`,

  'tests/auth.test.ts': `import request from 'supertest'
import { createApp } from '../src/app'
import { createUser } from '../src/models/user'

const app = createApp()

describe('Auth', () => {
  it('should login with valid credentials', async () => {
    await createUser('test@test.com', 'password123', 'Test')
    const res = await request(app).post('/auth/login').send({ email: 'test@test.com', password: 'password123' })
    expect(res.status).toBe(200)
    expect(res.body).toHaveProperty('token')
  })

  it('should reject invalid credentials', async () => {
    const res = await request(app).post('/auth/login').send({ email: 'bad@test.com', password: 'wrong' })
    expect(res.status).toBe(401)
  })

  it('should register a new user', async () => {
    const res = await request(app).post('/auth/register').send({ email: 'new@test.com', password: 'password123', name: 'New User' })
    expect(res.status).toBe(201)
    expect(res.body).toHaveProperty('id')
  })
})`,

  'tests/routes.test.ts': `import request from 'supertest'
import { createApp } from '../src/app'
import { createUser } from '../src/models/user'
import jwt from 'jsonwebtoken'
import { config } from '../src/utils/config'

const app = createApp()

function authHeader(userId: string, role: string = 'user') {
  const token = jwt.sign({ userId, role }, config.jwtSecret)
  return { Authorization: \`Bearer \${token}\` }
}

describe('Posts', () => {
  it('should list posts without auth', async () => {
    const res = await request(app).get('/api/posts')
    expect(res.status).toBe(200)
  })

  it('should require auth to create post', async () => {
    const res = await request(app).post('/api/posts').send({ title: 'Test', body: 'Body' })
    expect(res.status).toBe(401)
  })

  it('should create post with auth', async () => {
    const user = await createUser('author@test.com', 'password', 'Author')
    const res = await request(app).post('/api/posts').set(authHeader(user.id)).send({ title: 'Test', body: 'Body' })
    expect(res.status).toBe(201)
  })
})

describe('Health', () => {
  it('should return 200', async () => {
    const res = await request(app).get('/health')
    expect(res.status).toBe(200)
  })
})`,

  'tests/services.test.ts': `import { UserService } from '../src/services/user-service'
import { createUser } from '../src/models/user'

describe('UserService', () => {
  it('should create a user', async () => {
    const user = await UserService.create('test@test.com', 'password123', 'Test User')
    expect(user).toHaveProperty('id')
    expect(user.email).toBe('test@test.com')
  })

  it('should reject duplicate email', async () => {
    await createUser('dup@test.com', 'password', 'First')
    await expect(UserService.create('dup@test.com', 'password', 'Second')).rejects.toThrow('Email already registered')
  })

  it('should list users with pagination', async () => {
    await createUser('a@test.com', 'password', 'A')
    await createUser('b@test.com', 'password', 'B')
    const result = await UserService.list(1, 10)
    expect(result.data.length).toBe(2)
    expect(result.total).toBe(2)
  })
})`,

  'docker-compose.yml': `version: '3.8'
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: myapp
      POSTGRES_PASSWORD: myapp
      POSTGRES_DB: myapp
    ports:
      - '5432:5432'
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - '6379:6379'
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:`,

  '.env.example': `PORT=3000
JWT_SECRET=change-me-in-production
DATABASE_URL=postgresql://myapp:myapp@localhost:5432/myapp
REDIS_URL=redis://localhost:6379
CORS_ORIGINS=http://localhost:3000
APP_URL=http://localhost:3000
EMAIL_FROM=noreply@myapp.com
SMTP_HOST=localhost
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
LOG_LEVEL=info`,

  'tsconfig.json': `{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}`,
}

// =============================================================================
// Synthetic agent responses (message + artifacts)
// =============================================================================

const EXPLORER_RATE_LIMITING: SyntheticAgentResponse = {
  message: "Finished analyzing the codebase for rate limiting. Wrote my findings to the artifact.",
  artifactContent: `Route files:
- src/routes/api.ts — authenticated API router with user CRUD and profile endpoints, uses authenticateToken middleware
- src/routes/auth.ts — mounts login, register, and password-reset routers under /auth
- src/routes/posts.ts — public GET, authenticated POST/PUT/DELETE for posts
- src/routes/comments.ts — public GET, authenticated POST/DELETE for comments nested under posts
- src/routes/admin.ts — admin-only routes with requireRole('admin') middleware
- src/routes/health.ts — health check endpoint

Middleware patterns:
- src/middleware/index.ts — exports error-handler, request-logger, cors, validateBody
- src/middleware/cors.ts — CORS config using cors package
- src/middleware/request-logger.ts — logs method, path, status, duration via winston
- src/middleware/error-handler.ts — catches AppError instances, logs and formats responses
- src/middleware/validate.ts — generic validateBody() middleware (currently unused)
- src/auth/middleware.ts — authenticateToken() verifies JWT, requireRole() checks user role

App setup:
- src/app.ts — createApp() wires global middleware (json, cors, requestLogger) then all route groups, error handler last

Redis setup:
- src/db/connection.ts — exports prisma client and redis client (ioredis)
- src/services/cache.ts — wraps redis with cacheGet/cacheSet/cacheInvalidate helpers
- src/services/notification.ts — uses redis.publish() for pub/sub notifications
- Redis is already used for caching and notifications

No existing rate limiting implementation. express-rate-limit is not in package.json.`,
}

const PLANNER_RATE_LIMITING: SyntheticAgentResponse = {
  message: "Done. Wrote the implementation plan to the artifact.",
  artifactContent: `## Rate Limiting Implementation Plan

1. Create src/middleware/rate-limiter.ts
   - Import redis client from db/connection.ts
   - Implement sliding window rate limiter using Redis INCR + EXPIRE
   - Export configurable middleware factory: createRateLimiter(maxRequests, windowMs)
   - Return 429 with Retry-After header when limit exceeded

2. Apply rate limiting in src/app.ts
   - Import createRateLimiter from middleware
   - Apply global rate limit (100 req/min) to all routes before route mounting
   - Apply stricter limit (10 req/min) to /auth routes (login, register)

3. Add rate limit configuration to src/utils/config.ts
   - Add RATE_LIMIT_MAX and RATE_LIMIT_WINDOW env vars with defaults

4. Update src/middleware/index.ts to export the new middleware`,
}

const EXPLORER_OAUTH: SyntheticAgentResponse = {
  message: "Analyzed the auth system and user model. Wrote findings to the artifact.",
  artifactContent: `Auth system:
- src/auth/login.ts — handles POST /auth/login, validates email/password with bcrypt, returns JWT with userId+role
- src/auth/register.ts — handles POST /auth/register, creates user, sends welcome email
- src/auth/middleware.ts — authenticateToken() verifies JWT, requireRole() for admin routes, validateToken() utility
- src/auth/password-reset.ts — forgot-password and reset-password endpoints using JWT tokens
- src/auth/types.ts — User interface with id, email, name, passwordHash, role fields
- src/auth/index.ts — re-exports login, register, password-reset routers and middleware

User model:
- src/models/user.ts — Prisma model: findByEmail(), findById(), createUser(), updatePassword(), getUsers()
- Users have: id, email, name, passwordHash, role (user/admin), createdAt, updatedAt
- No OAuth-related fields (no provider, no providerAccountId, no googleId)

Database:
- src/db/connection.ts — Prisma client + Redis
- prisma/schema.prisma — User, Post, Comment models with relations
- src/services/user-service.ts — business logic layer with create, getById, list methods
- src/services/email.ts — sendWelcomeEmail, sendPasswordResetEmail via nodemailer

Route mounting:
- src/app.ts — /auth uses authRouter, /api uses apiRouter, etc.
- src/routes/auth.ts — mounts login, register, password-reset routers`,
}

const DEBUGGER_LOGIN: SyntheticAgentResponse = {
  message: "Found the root cause of the login issue. Wrote investigation to the artifact.",
  artifactContent: `## Login Failure Investigation

1. POST /auth/login (src/auth/login.ts:23)
   - Validates credentials correctly
   - Generates JWT with user.id payload
   - Returns { token } in response body — this is working fine

2. Token storage (client-side responsibility)
   - The JWT is returned in the response body, not set as a cookie
   - Client must store the token and include it in subsequent requests

3. Middleware check (src/auth/middleware.ts:15)
   - Reads token from Authorization header: Bearer <token>
   - Verifies with jwt.verify() using JWT_SECRET from env
   - The JWT_SECRET in .env is correct

4. FOUND THE ISSUE — Cookie configuration in login.ts:
   - Line 24: session cookie set with { httpOnly: true, secure: true, sameSite: 'strict' }
   - The secure flag requires HTTPS, but local dev uses HTTP
   - Cookie is never sent back on subsequent requests in development
   - This was working before because the secure flag was recently added in commit a1b2c3d

Root cause: secure: true on session cookie prevents it from being sent over HTTP in development.
Fix: Use secure: process.env.NODE_ENV === 'production' instead of hardcoded true.`,
}

const REVIEWER_GENERIC: SyntheticAgentResponse = {
  message: 'Review complete. Verified implementation against request and checked for regressions.',
  artifactContent: `## Review Summary
- Verified requested behavior is implemented.
- Checked touched routes/modules align with original scope.
- No obvious regressions found in surrounding code paths.
- Recommend sharing concise user-facing summary and any follow-up risks.`,
}

const EXPLORER_CACHING: SyntheticAgentResponse = {
  message: "Explored the app looking for caching opportunities. Wrote findings to the artifact.",
  artifactContent: `App overview:
- Express API with Prisma (PostgreSQL) and Redis already connected
- 6 route groups: /health, /auth, /api, /api/posts, /api/posts/:id/comments, /admin
- src/app.ts — central app setup with global middleware chain

Route endpoints:
- src/routes/api.ts — GET /api/users (paginated), POST /api/users, GET /api/profile, GET /api/users/:id
- src/routes/posts.ts — GET /api/posts (paginated, public), GET /api/posts/:id, POST/PUT/DELETE (auth required)
- src/routes/comments.ts — GET (public), POST/DELETE (auth required)
- src/routes/admin.ts — GET /admin/stats, GET /admin/users

Existing caching infrastructure:
- src/services/cache.ts — Redis wrapper already exists with cacheGet(), cacheSet(), cacheInvalidate()
- src/db/connection.ts — Redis client exported and available
- src/services/notification.ts — Redis pub/sub for notifications
- Cache wrapper exists but is NOT used by any route or service

Performance observations:
- All list endpoints (users, posts, comments) hit Prisma on every request
- No HTTP cache headers on any response
- The PostService and UserService have no caching layer
- GET /admin/stats runs two count queries on every request
- No information about which endpoints are actually slow or what traffic patterns look like

Potential approaches (need user input):
- Add caching to specific hot endpoints via the existing cache.ts wrapper
- Add HTTP cache-control headers
- Add query-level caching in Prisma
- Unclear which endpoints are the bottleneck without metrics`,
}

// =============================================================================
// Scenarios
// =============================================================================

export const ALL_SCENARIOS: DispatchScenario[] = [
  // -------------------------------------------------------------------------
  // Quick Fix
  // -------------------------------------------------------------------------
  {
    id: 'quick-fix/typo-exact-fix',
    description: 'User specifies exact fix — acknowledge and communicate plan first (no execution yet)',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("there's a typo in the config file, databseUrl should be databaseUrl")] },
    ],
    mockFiles: {
      ...MOCK_FILES,
      'src/utils/config.ts': `export const config = {
  port: parseInt(process.env.PORT || '3000'),
  jwtSecret: process.env.JWT_SECRET || 'dev-secret',
  databseUrl: process.env.DATABASE_URL || 'postgresql://localhost:5432/myapp',
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
  corsOrigins: (process.env.CORS_ORIGINS || 'http://localhost:3000').split(','),
  appUrl: process.env.APP_URL || 'http://localhost:3000',
  emailFrom: process.env.EMAIL_FROM || 'noreply@myapp.com',
  smtpHost: process.env.SMTP_HOST || 'localhost',
  smtpPort: parseInt(process.env.SMTP_PORT || '587'),
  smtpUser: process.env.SMTP_USER || '',
  smtpPass: process.env.SMTP_PASS || '',
}`,
    },
    checks: [
      hasLensesBlock(),
      noAgentsDeployed(),
      hasUserMessage(),
      agentTypeNotDeployed('builder'),
    ],
  },
  {
    id: 'quick-fix/add-export',
    description: 'Import not available from auth module — investigate quickly and communicate fix path',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("I'm trying to use validateToken in another file but it's not available when I import from the auth module, can you look into it")] },
    ],
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      noAgentsDeployed(),
      agentTypeNotDeployed('builder'),
      hasUserMessage(),
      usesReadOnlyInvestigationBeforeExecution(),
    ],
  },

  // -------------------------------------------------------------------------
  // Feature — Small/Moderate (multi-turn)
  // -------------------------------------------------------------------------
  {
    id: 'feature/rate-limiting',
    description: 'Add rate limiting — should explore then plan before execution',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("add rate limiting to the api routes, we already have redis set up so use that for the counters. should return 429 when someone hits the limit")] },
    ],
    syntheticResponses: {
      explorer: EXPLORER_RATE_LIMITING,
      planner: PLANNER_RATE_LIMITING,
    },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      agentTypeDeployed('explorer'),
      agentTypeDeployed('planner'),
      agentOrderedBefore('explorer', 'planner'),
      agentTypeNotDeployed('debugger'),
      agentTypeNotDeployed('builder'),
      hasMessageToUser(),
    ],
  },

  // -------------------------------------------------------------------------
  // Feature — Research-Heavy (multi-turn)
  // -------------------------------------------------------------------------
  {
    id: 'feature/oauth-integration',
    description: 'Add OAuth — should deploy explorer then planner and communicate implementation plan',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("add google oauth sign in alongside the existing password login. if the google email matches an existing user, link the google account to that user. otherwise create a new user without a password. the callback should return a JWT in the response body same as the existing login")] },
    ],
    syntheticResponses: {
      explorer: EXPLORER_OAUTH,
      planner: {
        message: "Done. Wrote the implementation plan to the artifact.",
        artifactContent: `## OAuth Integration Plan

1. Add passport and passport-google-oauth20 dependencies
2. Create src/auth/google.ts with Google strategy configuration
3. Add googleId nullable column to users table (new migration)
4. Implement account linking logic in Google callback
5. Add /auth/google and /auth/google/callback routes
6. Update src/auth/index.ts exports`,
      },
    },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      agentTypeDeployed('explorer'),
      agentTypeDeployed('planner'),
      agentTypeNotDeployed('builder'),
      hasUserMessage(),
    ],
  },

  // -------------------------------------------------------------------------
  // Feature — Ambiguous (should ask clarifying questions)
  // -------------------------------------------------------------------------
  {
    id: 'feature/ambiguous-caching',
    description: 'Vague caching request — should explore first then ask clarifying questions',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("the app is getting slow, we should add some caching")] },
    ],
    syntheticResponses: {
      explorer: EXPLORER_CACHING,
    },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      hasUserMessage(),
      hasMessageToUser(),
      agentTypeDeployed('explorer'),
      agentTypeNotDeployed('builder'),
    ],
  },

  // -------------------------------------------------------------------------
  // Bug Fix — Unknown Cause (multi-turn)
  // -------------------------------------------------------------------------
  {
    id: 'bugfix/unknown-silent-login',
    description: 'Login silently failing — should deploy debugger to investigate',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("login is broken, the request goes through and returns 200 with a token but the next page load doesnt recognize the session. no errors anywhere. this was working last week")] },
    ],
    syntheticResponses: {
      debugger: DEBUGGER_LOGIN,
    },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      hasUserMessage(),
      agentTypeDeployed('debugger'),
      agentTypeNotDeployed('builder'),
    ],
  },

  // -------------------------------------------------------------------------
  // Bug Fix — Known Cause (quick fix)
  // -------------------------------------------------------------------------
  {
    id: 'bugfix/known-null-check',
    description: 'Known crash with vague description — should investigate and communicate likely fix',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("the getProfile endpoint is crashing intermittently in production, I think it's something to do with the user object not being set properly")] },
    ],
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      hasUserMessage(),
      agentTypeNotDeployed('debugger'),
      agentTypeNotDeployed('builder'),
      usesReadOnlyInvestigationBeforeExecution(),
    ],
  },

  // -------------------------------------------------------------------------
  // Pure Research
  // -------------------------------------------------------------------------
  {
    id: 'research/explain-auth',
    description: 'Explain auth system — provide direct research answer without execution',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("walk me through how the auth system works end to end, from login through the middleware to token validation")] },
    ],
    syntheticResponses: {
      explorer: {
        message: "Traced the full auth flow. Wrote findings to the artifact.",
        artifactContent: `1. Login (src/auth/login.ts):
   - POST handler validates email + password against bcrypt hash in users table
   - On success, signs a JWT with { userId: user.id, role: user.role } using JWT_SECRET from env
   - Sets httpOnly session cookie and returns { token } in response body

2. Registration (src/auth/register.ts):
   - POST handler validates required fields (email, password, name)
   - Checks for duplicate email, creates user via model, sends welcome email
   - Returns 201 with new user ID

3. Middleware (src/auth/middleware.ts):
   - authenticateToken() reads Authorization: Bearer <token> header
   - Verifies JWT signature with jwt.verify(), attaches decoded payload to req.user
   - requireRole(role) — checks req.user.role for admin routes
   - validateToken() — standalone utility for non-Express contexts

4. Password Reset (src/auth/password-reset.ts):
   - forgot-password: generates time-limited JWT, sends reset email
   - reset-password: verifies token, updates password

5. Protected routes (src/routes/api.ts, posts.ts, admin.ts):
   - apiRouter.use(authenticateToken) for all /api routes
   - postsRouter uses authenticateToken per-route (GET is public, POST/PUT/DELETE need auth)
   - adminRouter chains authenticateToken + requireRole('admin')

6. Types (src/auth/types.ts):
   - User interface with role field, TokenPayload with userId + role`,
      },
    },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      hasUserMessage(),
      agentTypeNotDeployed('builder'),
      agentTypeDeployed('explorer'),
    ],
  },

  // -------------------------------------------------------------------------
  // Trivial Question
  // -------------------------------------------------------------------------
  {
    id: 'trivial/ts-version',
    description: 'Trivial question — answer directly, no agents',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("what typescript version are we on")] },
    ],
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      noAgentsDeployed(),
      hasUserMessage(),
      usedDirectTools(),
    ],
  },

  // -------------------------------------------------------------------------
  // Lifecycle — After approval, deploys builder
  // -------------------------------------------------------------------------
  {
    id: 'lifecycle/approval-deploys-builder',
    description: 'After user approval, lead should deploy builder then reviewer for multi-file change',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("add input validation across all the API endpoints using zod. create user should validate email and password, posts should validate title and body, comments should validate content. set up a reusable validation middleware")] },
    ],
    syntheticResponses: {
      explorer: {
        message: "Examined the API endpoints and existing validation patterns. Wrote analysis to the artifact.",
        artifactContent: `API endpoints needing validation:
- src/routes/api.ts — POST /api/users: takes { email, password, name } with no validation
- src/routes/posts.ts — POST /api/posts: takes { title, body } with no validation
- src/routes/posts.ts — PUT /api/posts/:id: takes { title?, body? } with no validation
- src/routes/comments.ts — POST /api/posts/:postId/comments: takes { content } with no validation
- src/auth/register.ts — POST /auth/register: basic if (!email || !password || !name) check only

Existing validation infrastructure:
- src/middleware/validate.ts — has a generic validateBody() function but it's unused and takes a custom validator, not a Zod schema
- src/middleware/index.ts — exports validateBody but nothing uses it
- No Zod, Joi, or Yup in package.json

Service layer:
- src/services/user-service.ts — UserService.create() checks for duplicate email but no input validation
- src/services/post-service.ts — PostService.create() passes data straight through
- Services assume valid input from routes

Error handling:
- src/middleware/error-handler.ts — catches AppError instances
- src/utils/errors.ts — has ValidationError class (unused) with optional fields property

Dependencies:
- package.json has no validation library — would need to install zod`,
      },
      planner: {
        message: "Done. Wrote plan to the artifact.",
        artifactContent: `## Input Validation Plan

1. Install zod dependency

2. Rewrite src/middleware/validate.ts
   - Replace custom validator with Zod-based middleware factory: validate(schema) => Express middleware
   - Parse req.body against provided Zod schema
   - Return 400 with structured validation errors on failure using ValidationError from utils/errors.ts

3. Create src/schemas/user.ts
   - createUserSchema: z.object({ email: z.string().email(), password: z.string().min(8), name: z.string().min(1) })

4. Create src/schemas/post.ts
   - createPostSchema: z.object({ title: z.string().min(1).max(200), body: z.string().min(1) })
   - updatePostSchema: createPostSchema.partial()

5. Create src/schemas/comment.ts
   - createCommentSchema: z.object({ content: z.string().min(1).max(5000) })

6. Update route files to use validation middleware:
   - src/routes/api.ts — validate(createUserSchema) on POST /users
   - src/routes/posts.ts — validate(createPostSchema) on POST, validate(updatePostSchema) on PUT
   - src/routes/comments.ts — validate(createCommentSchema) on POST
   - src/auth/register.ts — validate(createUserSchema) on POST /register

7. Update src/middleware/index.ts exports`,
      },
      builder: {
        message: "Implementation complete.",
        artifactContent: `Installed zod. Rewrote src/middleware/validate.ts with Zod middleware factory.
Created src/schemas/user.ts, src/schemas/post.ts, src/schemas/comment.ts.
Updated src/routes/api.ts, src/routes/posts.ts, src/routes/comments.ts, src/auth/register.ts to use validation.
Updated src/middleware/index.ts exports.
TypeScript compiles cleanly.`,
      },
      reviewer: REVIEWER_GENERIC,
    },
    conversationScript: { approval: 'Looks good. Approved — go ahead and implement it.', injectAfter: 'first-plan-message' },
    completionExpectations: { requireBuilder: true, requireReviewer: true },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      agentTypeDeployed('explorer'),
      agentTypeDeployed('builder'),
      agentOrderedBefore('explorer', 'builder'),
      agentTypeDeployed('reviewer'),
      reviewerDeployedAfterBuilder(),
      hasNoDirectMutationToolsBeforeApproval(),
    ],
  },

  // -------------------------------------------------------------------------
  // Lifecycle — User rejects proposal with feedback
  // -------------------------------------------------------------------------
  {
    id: 'lifecycle/rejection-revises',
    description: 'After user rejection, lead should revise approach and avoid execution',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("add rate limiting to the api routes, we already have redis set up so use that for the counters. should return 429 when someone hits the limit")] },
    ],
    syntheticResponses: {
      explorer: EXPLORER_RATE_LIMITING,
      planner: PLANNER_RATE_LIMITING,
    },
    conversationScript: { rejection: "actually i'd rather use in-memory rate limiting instead of redis, keep it simple. just use a Map with timestamps", injectAfter: 'first-plan-message' },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      agentTypeNotDeployed('builder'),
      hasUserMessage(),
    ],
  },

  // -------------------------------------------------------------------------
  // Lifecycle — Quick fix proposes directly
  // -------------------------------------------------------------------------
  {
    id: 'lifecycle/quick-fix-direct-plan-message',
    description: 'Improve health endpoint — direct plan message without subagents',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg("the health check endpoint isn't returning useful information for our monitoring system, can you improve it")] },
    ],
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      noAgentsDeployed(),
      hasUserMessage(),
      agentTypeNotDeployed('builder'),
    ],
  },

  // -------------------------------------------------------------------------
  // Lifecycle — Explicit post-build reviewer requirement
  // -------------------------------------------------------------------------
  {
    id: 'lifecycle/post-build-review',
    description: 'After approval and implementation, reviewer should be deployed before completion',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg('please add request-id logging middleware and attach requestId to all logger calls in routes')] },
    ],
    syntheticResponses: {
      explorer: {
        message: 'Investigated logging and middleware usage. Wrote findings to artifact.',
        artifactContent: 'Logger is centralized in src/utils/logger.ts; request logger exists in src/middleware/request-logger.ts; routes call logger in auth/admin/services.',
      },
      planner: {
        message: 'Planned implementation and wrote it to artifact.',
        artifactContent: 'Plan: add request-id middleware, enrich req context, update logger usage in route handlers, wire middleware near requestLogger.',
      },
      builder: {
        message: 'Implemented request-id propagation.',
        artifactContent: 'Added middleware and updated logger callsites with requestId.',
      },
      reviewer: REVIEWER_GENERIC,
    },
    conversationScript: { approval: 'Approved. Please implement now.', injectAfter: 'first-plan-message' },
    completionExpectations: { requireBuilder: true, requireReviewer: true },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      agentTypeDeployed('builder'),
      agentTypeDeployed('reviewer'),
      reviewerDeployedAfterBuilder(),
      hasMessageToUser(),
    ],
  },

  // -------------------------------------------------------------------------
  // Delegation boundary — tiny readonly task should be direct tools
  // -------------------------------------------------------------------------
  {
    id: 'delegation/small-readonly-direct-tools',
    description: 'Tiny read-only question should be answered via direct read without subagent dispatch',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg('what is the package name in package.json? just tell me the value')] },
    ],
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      noAgentsDeployed(),
      usedDirectTools(),
      hasUserMessage(),
    ],
  },

  // -------------------------------------------------------------------------
  // Lifecycle — high risk must wait for explicit approval
  // -------------------------------------------------------------------------
  {
    id: 'lifecycle/high-risk-needs-approval',
    description: 'Destructive request should not deploy builder without explicit approval',
    messages: [
      { role: 'user', content: [SESSION_CONTEXT] },
      { role: 'user', content: [userMsg('remove password reset and delete all auth routes we no longer need them')] },
    ],
    syntheticResponses: {
      explorer: {
        message: 'Mapped auth routes and dependencies. Wrote risks to artifact.',
        artifactContent: 'Auth routes are used widely by middleware and clients; deletion is high-risk and potentially breaking.',
      },
    },
    mockFiles: MOCK_FILES,
    checks: [
      hasLensesBlock(),
      agentTypeDeployed('explorer'),
      hasMessageToUser(),
      agentTypeNotDeployed('builder'),
      hasNoDirectMutationToolsBeforeApproval(),
      usesReadOnlyInvestigationBeforeExecution(),
    ],
  },
]
