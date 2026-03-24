import { Context, Effect } from 'effect'
import type { WorkflowCriteriaState } from '../projections/workflow'

export interface WorkflowStateReader {
  readonly getState: (forkId: string | null) => Effect.Effect<WorkflowCriteriaState>
}

export class WorkflowStateReaderTag extends Context.Tag('WorkflowStateReader')<
  WorkflowStateReaderTag,
  WorkflowStateReader
>() {}