export const makeResidentWatchRegistry = <Resource>() => {
  const watches = new WeakMap<object, Map<Resource, unknown>>()
  return {
    getOrCreate: <Watch>(client: object, resource: Resource, create: () => Watch): Watch => {
      const clientWatches = watches.get(client) ?? new Map<Resource, unknown>()
      if (!watches.has(client)) watches.set(client, clientWatches)
      const existing = clientWatches.get(resource)
      if (existing !== undefined) return existing as Watch
      const watch = create()
      clientWatches.set(resource, watch)
      return watch
    },
  }
}
