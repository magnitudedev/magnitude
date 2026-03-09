# Task Manager Mock Project

A simple full-stack task manager fixture built with:

- **Server:** Bun + Elysia + Drizzle ORM + SQLite (`bun:sqlite`)
- **Client:** React + Vite

## Features

- User registration and login with JWT
- Create and manage projects
- Add/remove project members
- Create, assign, and update tasks by project

## Setup

```bash
bun install
bun run dev
```

Server runs on `http://localhost:3000`, client on Vite default port.

## Database

From `server/`:

```bash
bun run db:generate
bun run db:migrate
```