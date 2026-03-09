export const config = {
  port: Number(Bun.env.PORT ?? 3000),
  jwtSecret: Bun.env.JWT_SECRET ?? 'dev-secret',
  dbPath: Bun.env.DB_PATH ?? './task-manager.db',
}