export type SubmitDispatchEvent =
  | { type: 'interrupt'; forkId: string }
  | { type: 'user_message'; forkId: string | null }

export function buildSubmitDispatchEvents(selectedForkId: string | null): SubmitDispatchEvent[] {
  if (selectedForkId == null) {
    return [{ type: 'user_message', forkId: null }]
  }

  return [
    { type: 'interrupt', forkId: selectedForkId },
    { type: 'user_message', forkId: selectedForkId },
  ]
}

export function shouldHandleSlashCommandInTab(selectedForkId: string | null): boolean {
  return selectedForkId == null
}