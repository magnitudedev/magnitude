import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent, PhaseVerdictEntry } from '../events'
import { WorkflowProjection } from '../projections/workflow'
import { SessionContextProjection } from '../projections/session-context'
import { ExecutionManager } from '../execution/execution-manager'
import { getCurrentPhase, parseSkill } from '@magnitudedev/skills'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { createId, createShortId } from '../util/id'
import { agentEnv } from '../util/agent-env'
import { formatPhasePrompt } from '../prompts/skills'

export const WorkflowWorker = Worker.define<AppEvent>()({
  name: 'WorkflowWorker',

  eventHandlers: {
    skill_activated: (event, publish, read) => Effect.gen(function* () {
      const content = yield* Effect.tryPromise({
        try: () => readFile(event.skillPath, 'utf8'),
        catch: (e) => new Error(`Failed reading skill at ${event.skillPath}: ${e instanceof Error ? e.message : String(e)}`),
      }).pipe(Effect.orDie)

      const parsed = parseSkill(content)

      if (parsed.phases.length > 0) {
        const onStart = parsed.phases[0]?.hooks?.onStart
        if (onStart?.trim()) {
          const sessionCtx = yield* read(SessionContextProjection)
          const { cwd, workspacePath } = sessionCtx.context!
          yield* Effect.promise(() => Bun.spawn(['bash', '-lc', onStart], { cwd, env: agentEnv(cwd, workspacePath) }).exited)
        }
      }

      yield* publish({
        type: 'skill_started',
        forkId: event.forkId,
        source: event.source,
        skill: parsed,
      })
    }),

    phase_submitted: (event, publish, read) => Effect.gen(function* () {
      const workflow = yield* read(WorkflowProjection)
      const state = workflow.workflowState
      if (!state) return

      const phase = getCurrentPhase(state)
      if (!phase) return

      const sessionCtx = yield* read(SessionContextProjection)
      const { cwd, workspacePath } = sessionCtx.context!
      const runShell = (command: string) =>
        Effect.promise(() => Bun.spawn(['bash', '-lc', command], { cwd, env: agentEnv(cwd, workspacePath) }).exited)

      if (phase.hooks?.onSubmit) yield* runShell(phase.hooks.onSubmit)

      const criteria = phase.criteria ?? []
      if (criteria.length === 0) {
        if (phase.hooks?.onAccept) yield* runShell(phase.hooks.onAccept)

        const nextIndex = state.currentPhaseIndex + 1
        const nextPhase = state.skill.phases[nextIndex]
        if (nextPhase?.hooks?.onStart) yield* runShell(nextPhase.hooks.onStart)

        const workflowCompleted = nextIndex >= state.skill.phases.length
        yield* publish({
          type: 'phase_verdict',
          forkId: event.forkId,
          passed: true,
          verdicts: [],
          nextPhasePrompt: nextPhase ? formatPhasePrompt(nextPhase, nextIndex) : null,
          workflowCompleted,
        })

        if (workflowCompleted) {
          yield* publish({
            type: 'skill_completed',
            forkId: event.forkId,
            skillName: state.skill.name,
          })
        }
        return
      }

      yield* publish({
        type: 'phase_criteria_started',
        forkId: event.forkId,
        criteria: criteria.map((c, i) => ({
          index: i,
          name: c.name,
          type: c.type === 'shell-succeed' ? 'shell' as const : c.type === 'agent-approval' ? 'agent' as const : 'user' as const,
        })),
      })

      for (let index = 0; index < criteria.length; index++) {
        const criterion = criteria[index]

        if (criterion.type === 'user-approval') {
          yield* publish({
            type: 'phase_criteria_verdict',
            forkId: event.forkId,
            parentForkId: event.forkId,
            criteriaIndex: index,
            criteriaName: criterion.name,
            criteriaType: 'user',
            status: 'passed',
            reason: 'Auto-approved',
          } as AppEvent)
          continue
        }

        if (criterion.type !== 'shell-succeed') continue

        const proc = Bun.spawn(['bash', '-lc', criterion.command], { cwd, env: agentEnv(cwd, workspacePath) })
        const criteriaIndex = index
        const criteriaName = criterion.name
        const criteriaCommand = criterion.command
        yield* publish({
          type: 'phase_criteria_verdict',
          forkId: event.forkId,
          parentForkId: event.forkId,
          criteriaIndex,
          criteriaName,
          criteriaType: 'shell',
          status: 'running',
          command: criteriaCommand,
          pid: proc.pid,
        })

        proc.exited.then(async (exitCode) => {
          if (exitCode === 0) {
            await Effect.runPromise(publish({
              type: 'phase_criteria_verdict',
              forkId: event.forkId,
              parentForkId: event.forkId,
              criteriaIndex,
              criteriaName,
              criteriaType: 'shell',
              status: 'passed',
              command: criteriaCommand,
            }))
          } else {
            const stderrText = await new Response(proc.stderr).text().catch(() => '')
            const wasTruncated = stderrText.length > 500
            const truncated = wasTruncated ? stderrText.slice(-500) : stderrText
            let logNote = ''
            if (wasTruncated) {
              const logDir = `${workspacePath}/results`
              await mkdir(logDir, { recursive: true })
              const logId = createShortId()
              const logPath = `${logDir}/${logId}.log`
              await writeFile(logPath, stderrText)
              logNote = `\nFull output: ${logPath}`
            }
            const reason = truncated.trim()
              ? `Exit code ${exitCode}\n${truncated.trim()}${logNote}`
              : `Exit code ${exitCode}${logNote}`
            await Effect.runPromise(publish({
              type: 'phase_criteria_verdict',
              forkId: event.forkId,
              parentForkId: event.forkId,
              criteriaIndex,
              criteriaName,
              criteriaType: 'shell',
              status: 'failed',
              command: criteriaCommand,
              reason,
            }))
          }
        })
      }
    }),
  },

  signalHandlers: (on) => [
    on(WorkflowProjection.signals.shellCriteriaPassed, ({ forkId }, publish, read) => Effect.gen(function* () {
      const workflow = yield* read(WorkflowProjection)
      const workflowState = workflow.workflowState
      if (!workflowState) return
      const exec = yield* ExecutionManager
      const phase = getCurrentPhase(workflowState)
      if (!phase) return

      const criteria = phase.criteria ?? []
      for (let index = 0; index < criteria.length; index++) {
        const criterion = criteria[index]
        if (criterion.type !== 'agent-approval') continue

        const agentId = `reviewer-workflow-${createId()}`
        const spawnedForkId = yield* exec.fork({
          parentForkId: forkId,
          name: `Reviewer (${criterion.name})`,
          agentId,
          prompt: criterion.prompt,
          mode: 'spawn',
          role: 'reviewer',
          taskId: createId(),
          message: `${criterion.prompt}\n\nReport verdict with <phase-verdict passed="true|false" reason="..."/>.`,
        })

        yield* publish({
          type: 'phase_criteria_verdict',
          forkId: spawnedForkId,
          parentForkId: forkId,
          criteriaIndex: index,
          criteriaName: criterion.name,
          criteriaType: 'agent',
          status: 'running',
          agentId: spawnedForkId,
        })
      }
    })),

    on(WorkflowProjection.signals.phaseResolved, ({ forkId, passed, verdicts }, publish, read) => Effect.gen(function* () {
      const workflow = yield* read(WorkflowProjection)
      const workflowState = workflow.workflowState
      if (!workflowState) return
      const phase = getCurrentPhase(workflowState)
      if (!phase) return

      const sessionCtx = yield* read(SessionContextProjection)
      const { cwd, workspacePath } = sessionCtx.context!
      const runShell = (command: string) =>
        Effect.promise(() => Bun.spawn(['bash', '-lc', command], { cwd, env: agentEnv(cwd, workspacePath) }).exited)

      let nextPhasePrompt: string | null = null
      let workflowCompleted = false

      if (passed) {
        if (phase.hooks?.onAccept) yield* runShell(phase.hooks.onAccept)
        const nextIndex = workflowState.currentPhaseIndex + 1

        if (nextIndex >= workflowState.skill.phases.length) {
          workflowCompleted = true
        } else {
          const nextPhase = workflowState.skill.phases[nextIndex]
          if (nextPhase?.hooks?.onStart) yield* runShell(nextPhase.hooks.onStart)
          nextPhasePrompt = nextPhase ? formatPhasePrompt(nextPhase, nextIndex) : null
        }
      } else {
        if (phase.hooks?.onReject) yield* runShell(phase.hooks.onReject)
      }

      const normalizedVerdicts: readonly PhaseVerdictEntry[] = verdicts.map((v) => ({
        criteriaIndex: v.criteriaIndex,
        criteriaName: v.criteriaName,
        passed: v.passed,
        reason: v.reason,
      }))

      yield* publish({
        type: 'phase_verdict',
        forkId,
        passed,
        verdicts: normalizedVerdicts,
        nextPhasePrompt,
        workflowCompleted,
      })

      if (workflowCompleted) {
        yield* publish({
          type: 'skill_completed',
          forkId,
          skillName: workflowState.skill.name,
        })
      }
    })),
  ],
})
