export const hasConversationActivity = ({
  displayMessageCount,
  bashOutputCount,
}: {
  displayMessageCount: number
  bashOutputCount: number
}): boolean => displayMessageCount > 0 || bashOutputCount > 0
