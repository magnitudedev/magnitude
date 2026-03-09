import { Elysia } from 'elysia'
import { cors } from '@elysiajs/cors'
import { jwt } from '@elysiajs/jwt'
import { authRoutes } from './routes/auth'
import { healthRoutes } from './routes/health'
import { projectRoutes } from './routes/projects'
import { taskRoutes } from './routes/tasks'
import { config } from './utils/config'
import { logger } from './utils/logger'

export const app = new Elysia()
  .use(cors())
  .use(
    jwt({
      name: 'jwt',
      secret: config.jwtSecret,
    }),
  )
  .use(healthRoutes)
  .use(authRoutes)
  .use(projectRoutes)
  .use(taskRoutes)

app.listen(config.port)
logger.info(`Task manager server running on http://localhost:${config.port}`)