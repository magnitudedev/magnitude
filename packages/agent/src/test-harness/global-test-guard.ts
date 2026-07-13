const queue: Array<() => void> = []
let locked = false

const acquire = async (): Promise<() => void> => {
  if (!locked) {
    locked = true
    return () => release()
  }

  await new Promise<void>((resolve) => {
    queue.push(resolve)
  })

  locked = true
  return () => release()
}

const release = (): void => {
  const next = queue.shift()
  if (next) {
    next()
    return
  }
  locked = false
}

export async function runWithGlobalAgentTestGuard<T>(
  _name: string,
  fn: () => Promise<T> | T
): Promise<T> {
  const unlock = await acquire()
  try {
    return await fn()
  } finally {
    unlock()
  }
}