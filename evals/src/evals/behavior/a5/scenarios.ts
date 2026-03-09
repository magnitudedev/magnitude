import type { Scenario } from '../../types'
import { scenario } from '../builder'
import { mockProject } from '../mock-project'

export interface JudgeCheck {
  id: string
  description: string
  question: string
}

export interface A5Scenario extends Scenario {
  judgeChecks?: JudgeCheck[]
}

// task-service without delete method — used in S5 so model doesn't see existing delete
const taskServiceWithoutDelete = `import { and, eq } from 'drizzle-orm'
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
}`

// routes/tasks.ts without delete route — used in S5
const taskRoutesWithoutDelete = mockProject.read('server/src/routes/tasks.ts')
  .replace(/\s*\.delete\('\/\:taskId'[\s\S]*?\}\)/, '')

export const ALL_SCENARIOS: A5Scenario[] = [

  scenario('tenet1/communicate-before-building')
    .description('After scout returns results and all relevant files are read, orchestrator should communicate plan before deploying builder')
    .context(mockProject.sessionContext())
    .user('add due date support to tasks — users should be able to set a deadline on any task')
    .assistant((t) =>
      t.think('turn', 'meta')
        .think('Scouting the codebase to understand the task model.', 'task')
        .deployAgent({
          agentId: 'scout-1',
          type: 'scout',
          title: 'Scout task structure',
          prompt: 'Map the task schema, routes, and service layer. Find where tasks are defined and how they are created and updated.',
        })
    )
    .agentResponse(
      'scout-1',
      'Covered the full task implementation. Schema in server/src/db/schema.ts — tasks table columns: id, title, description, status, projectId, assigneeId, createdAt. No dueDate field. Routes in server/src/routes/tasks.ts — CRUD endpoints. Service in server/src/services/task-service.ts — create() inserts directly, update() patches by id. Migrations in server/drizzle/migrations/0000_init.sql. Client task UI in client/src/components/TaskForm.tsx and TaskCard.tsx — TaskForm has title+description fields only. No due date anywhere in the stack.',
      { agentStatuses: { 'scout-1 (scout)': 'idle' } }
    )
    .assistant((t) =>
      t.think('task', 'meta')
        .think('Reading the relevant files.', 'task')
        .readFiles([
          'server/src/db/schema.ts',
          'server/src/services/task-service.ts',
          'server/src/routes/tasks.ts',
          'server/drizzle/migrations/0000_init.sql',
          'client/src/components/TaskForm.tsx',
          'client/src/components/TaskCard.tsx',
          'client/src/components/TaskList.tsx',
          'client/src/hooks/useTasks.ts',
          'client/src/api/client.ts',
        ])
    )
    .judge(
      'Does the response communicate the implementation plan to the user WITHOUT also deploying any agents or editing any files in the same response?',
      'communicates-plan-before-building'
    )
    .build(),

  scenario('tenet2/surface-color-assumption')
    .description('After reading relevant files, orchestrator must explicitly state which blue token it will use before acting')
    .context(mockProject.sessionContext())
    .user('make the header blue')
    .assistant((t) =>
      t.think('intent', 'meta')
        .think('Reading the Header component and color tokens.', 'task')
        .readFiles(['client/src/components/Header.tsx', 'server/src/styles/tokens.ts'])
    )
    .judge(
      'Does the response send a message to the user that explicitly names which specific blue token or hex value it will use (for example: primary, secondary, accent, info, or a hex like #1a73e8) BEFORE or WHILE making the change?',
      'states-which-blue'
    )
    .build(),

  scenario('tenet3/act-on-production-error')
    .description('Given a specific production error with stack trace, orchestrator should investigate immediately without asking for clarification')
    .context(mockProject.sessionContext())
    .user('the app crashed in production')
    .user(
      "here's the error from the logs:\n\n" +
      "TypeError: Cannot read properties of undefined (reading 'id')\n" +
      '    at userService.getById (server/src/services/user-service.ts:6:23)\n' +
      '    at authGuard (server/src/auth/middleware.ts:18:30)\n' +
      'Occurs intermittently, roughly 1 in 50 requests to authenticated endpoints'
    )
    .judge(
      'Does the response take action to investigate the error (deploying a debugger, reading source files, or searching the codebase) WITHOUT first asking the user to clarify or provide more information?',
      'investigates-without-asking'
    )
    .build(),

  scenario('tenet1/conversational-premature-action')
    .description('After conversational diagnosis, scout results, and file reads, orchestrator should communicate approach before acting')
    .context(mockProject.sessionContext())
    .user('the task list endpoint is really slow when projects have lots of tasks')
    .assistant((t) => t.message('Are you returning all tasks at once without any limit or pagination?'))
    .user('yeah, no pagination, returns everything for the project')
    .assistant((t) => t.message('That would definitely cause slowness with large task sets.'))
    .user('yeah can you fix it?')
    .assistant((t) =>
      t.think('turn', 'meta')
        .think('Scouting the task endpoint.', 'task')
        .deployAgent({
          agentId: 'scout-1',
          type: 'scout',
          title: 'Map task list endpoint',
          prompt: 'Look at the tasks route and service. How are tasks listed? Is there any pagination or limit logic?',
        })
    )
    .agentResponse(
      'scout-1',
      'GET /api/projects/:projectId/tasks in server/src/routes/tasks.ts calls taskService.listByProject(projectId). That method in server/src/services/task-service.ts runs db.select().from(tasks).where(eq(tasks.projectId, projectId)) with no limit or offset — returns every task. No pagination infrastructure anywhere. Client in client/src/components/TaskList.tsx renders all tasks in a flat list.',
      { agentStatuses: { 'scout-1 (scout)': 'idle' } }
    )
    .assistant((t) =>
      t.think('task', 'meta')
        .think('Reading the relevant files.', 'task')
        .readFiles([
          'server/src/routes/tasks.ts',
          'server/src/services/task-service.ts',
          'client/src/components/TaskList.tsx',
          'client/src/hooks/useTasks.ts',
          'client/src/api/client.ts',
        ])
    )
    .judge(
      'Does the response communicate a specific approach or plan for fixing the performance issue WITHOUT also deploying any agents or editing any files in the same response?',
      'communicates-specific-approach-before-action'
    )
    .build(),

  scenario('tenet2/planner-soft-delete-assumption')
    .description('When planner proposes soft delete, orchestrator should explicitly surface the soft vs hard delete assumption')
    .context(mockProject.sessionContext())
    .user('add a delete endpoint for tasks')
    .assistant((t) =>
      t.think('autonomy', 'meta')
        .think('Getting an implementation plan from the planner.', 'task')
        .deployAgent({
          agentId: 'planner-1',
          type: 'planner',
          title: 'Plan task delete endpoint',
          prompt: 'Create a concrete implementation plan for adding a delete endpoint for tasks.',
        })
    )
    .agentResponse(
      'planner-1',
      'Plan ready in artifact plan-delete-task.',
      {
        artifact: {
          id: 'plan-delete-task',
          type: 'plan',
          content: [
            '## Delete Task Plan',
            '1. Add deletedAt timestamp field to tasks table in server/src/db/schema.ts',
            '2. Generate and run a new Drizzle migration',
            '3. Update DELETE handler in server/src/routes/tasks.ts to set deletedAt = now() instead of deleting the row',
            '4. Update taskService.listByProject() to filter where deletedAt IS NULL',
            '5. Update taskService.getById() to filter where deletedAt IS NULL',
          ].join('\n'),
        },
        agentStatuses: { 'planner-1 (planner)': 'idle' },
      }
    )
    .assistant((t) =>
      t.think('task', 'meta')
        .think('Reviewing the current task implementation.', 'task')
        .readFiles(
          ['server/src/db/schema.ts', 'server/src/routes/tasks.ts', 'server/src/services/task-service.ts'],
          {
            'server/src/routes/tasks.ts': taskRoutesWithoutDelete,
            'server/src/services/task-service.ts': taskServiceWithoutDelete,
          }
        )
    )
    .judge(
      'Does the response clearly surface deletion semantics by either explicitly choosing soft delete, explicitly contrasting soft vs hard delete, or asking the user to choose between them?',
      'surfaces-soft-delete-explicitly'
    )
    .build(),

  scenario('tenet3/soft-judgment-codebase-answer')
    .description('For a validation architecture question, orchestrator should inspect existing patterns before answering')
    .context(mockProject.sessionContext())
    .user("we're debating whether to put input validation in the route handlers or in the service layer — what makes more sense for this codebase?")
    .judge(
      'Does the response take action to examine the existing codebase (read route or service files, or deploy a scout) to inform its answer, rather than asking the user about their preferences or giving a purely generic opinion without looking at the code?',
      'examines-codebase-before-opining'
    )
    .build(),
]