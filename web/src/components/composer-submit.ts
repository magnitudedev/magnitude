export function routeComposerSubmission(options: {
  readonly text: string
  readonly trySlashCommand: ((text: string) => boolean) | undefined
  readonly onHandledCommand: () => void
  readonly onMessage: () => void
}): 'command' | 'message' {
  if (options.trySlashCommand?.(options.text)) {
    options.onHandledCommand()
    return 'command'
  }
  options.onMessage()
  return 'message'
}
