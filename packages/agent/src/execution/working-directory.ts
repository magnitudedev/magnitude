import { Context } from 'effect'

export interface WorkingDirectoryService {
  readonly cwd: string
}

export class WorkingDirectoryTag extends Context.Tag('WorkingDirectory')<
  WorkingDirectoryTag,
  WorkingDirectoryService
>() {}