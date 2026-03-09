import type { Scenario } from '../../../types'
import { scenario } from '../builder'
import { mockProject } from '../mock-project'

export interface JudgeCheck {
  id: string
  description: string
  question: string
}

export interface ToolUsageScenario extends Scenario {
  judgeChecks?: JudgeCheck[]
}

export const ALL_SCENARIOS: ToolUsageScenario[] = [

  scenario('tool-usage/scout-vs-reads')
    .description('When asked to add a feature to an unfamiliar area, orchestrator should deploy scout rather than reading files itself')
    .context(mockProject.sessionContext())
    .user('add request logging to all routes — log method, path, status code, and duration for every request')
    .judge(
      'Does the response deploy a scout agent to explore the codebase rather than immediately reading individual files itself using fs-read or similar tools?',
      'deploys-scout-not-reads'
    )
    .build(),

  scenario('tool-usage/builder-vs-edits')
    .description('After user approves a multi-file plan, orchestrator should deploy builder rather than editing files itself')
    .context(mockProject.sessionContext())
    .user('add role-based access control — project owners can delete any task, members can only delete tasks they created')
    .assistant((t) =>
      t.think('Getting an implementation plan from the planner.')
        .deployAgent({
          agentId: 'planner-1',
          type: 'planner',
          title: 'Plan RBAC for task deletion',
          prompt: 'Plan role-based access control for task deletion. Project owners can delete any task, members can only delete tasks they created.',
        })
    )
    .agentResponse(
      'planner-1',
      'Plan ready in artifact rbac-plan.',
      {
        artifact: {
          id: 'rbac-plan',
          type: 'plan',
          content: [
            '## RBAC Plan',
            '1. Add createdBy field to tasks table in server/src/db/schema.ts',
            '2. Generate and run a new Drizzle migration',
            '3. Update taskService.create() to accept and store createdBy',
            '4. Update DELETE handler in server/src/routes/tasks.ts to check: if user is project owner allow; else only allow if task.createdBy === user.id',
            '5. Add test cases to server/tests/tasks.test.ts',
          ].join('\n'),
        },
        agentStatuses: { 'planner-1 (planner)': 'idle' },
      }
    )
    .assistant((t) =>
      t.think('Reviewing the relevant files.')
        .readFiles([
          'server/src/db/schema.ts',
          'server/src/routes/tasks.ts',
          'server/src/services/task-service.ts',
          'server/src/services/project-service.ts',
          'server/tests/tasks.test.ts',
        ])
    )
    .assistant((t) => t.message("Here's the plan for adding RBAC to task deletion. Want me to go ahead?"))
    .user('yeah go for it')
    .judge(
      'Does the response deploy a builder agent to implement the plan rather than making file edits directly using edit, fs-write, or shell commands?',
      'deploys-builder-not-edits'
    )
    .build(),

  scenario('tool-usage/planner-vs-self-plan')
    .description('When asked to plan a feature with real tradeoffs, orchestrator should deploy planner rather than drafting the plan itself')
    .context(mockProject.sessionContext())
    .user('we need to add notifications — users should get notified when a task is assigned to them. can you plan this out? we could do polling, websockets, or webhooks to an external service, not sure which fits best')
    .judge(
      'Does the response deploy a planner agent to evaluate the tradeoffs and produce the plan, rather than writing out the plan directly itself?',
      'deploys-planner-not-self-plans'
    )
    .build(),

  scenario('tool-usage/researcher-vs-scout')
    .description('For a deep end-to-end question about a specific system, orchestrator should deploy researcher rather than scout')
    .context(mockProject.sessionContext())
    .user('how does auth token validation work end-to-end in this codebase? i want to understand exactly what happens from the moment a request comes in to when we know if the user is authenticated')
    .judge(
      'Does the response deploy a researcher agent rather than a scout agent to investigate the auth token validation flow?',
      'deploys-researcher-not-scout'
    )
    .build(),
]