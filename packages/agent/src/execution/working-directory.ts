import { Context } from 'effect'

export interface WorkingDirectoryService {
  readonly cwd: string
  readonly scratchpadPath: string
}

export class WorkingDirectoryTag extends Context.Tag('WorkingDirectory')<
  WorkingDirectoryTag,
  WorkingDirectoryService
>() {}