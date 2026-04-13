import { Effect } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { SessionContextProjection } from '../projections/session-context'
import { parseSkill } from '@magnitudedev/skills'
import { readFile } from 'node:fs/promises'
import { agentEnv } from '../util/agent-env'

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
  },
})
