import { Context } from 'effect'

export interface ToolExecutionContext {
  readonly turnId: string
}

export class ToolExecutionContextTag extends Context.Tag('ToolExecutionContext')<
  ToolExecutionContextTag,
  ToolExecutionContext
>() {}