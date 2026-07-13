const makeDraftOwnerId = (): string =>
  `draft-owner-${crypto.randomUUID()}`

const draftOwnerId = makeDraftOwnerId()

export const getDraftSessionOwnerId = (): string => draftOwnerId
