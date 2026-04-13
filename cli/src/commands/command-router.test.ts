import { describe, expect, mock, test } from 'bun:test'
import { routeSlashCommand, type CommandContext } from './command-router'

function createContext(overrides: Partial<CommandContext> = {}): CommandContext {
  return {
    resetConversation: mock(() => {}),
    showSystemMessage: mock(() => {}),
    exitApp: mock(() => {}),
    openRecentChats: mock(() => {}),
    enterBashMode: mock(() => {}),
    activateSkill: mock(() => {}),
    initProject: mock(() => {}),
    openSettings: mock(() => {}),
    openSetup: mock(() => {}),
    openBrowserSetup: mock(() => {}),
    ...overrides,
  }
}

describe('routeSlashCommand', () => {
  test('handles recognized commands', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/new', ctx)).toBe(true)
    expect(ctx.resetConversation).toHaveBeenCalledTimes(1)
  })

  test('unknown command is not handled', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/definitely-not-a-command', ctx)).toBe(false)
    expect(ctx.showSystemMessage).not.toHaveBeenCalled()
  })

  test('slash-prefixed filesystem-like text is not handled', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/Users/me/a.png /Users/me/b.png', ctx)).toBe(false)
    expect(routeSlashCommand('/home/me/a.png /home/me/b.png', ctx)).toBe(false)
    expect(ctx.showSystemMessage).not.toHaveBeenCalled()
  })
})
