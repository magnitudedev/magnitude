import { useState, useEffect, useCallback } from 'react'
import type { KeyEvent } from '@opentui/core'
import { Effect } from 'effect'
import { EmptySkillset, SkillsetResolver, SkillsetResolverLive } from '@magnitudedev/skills'
import type { Skillset, SkillsetInfo } from '@magnitudedev/skills'
import { useStorage } from '../providers/storage-provider'

export interface UseSkillsetSettingsResult {
  availableSkillsets: SkillsetInfo[]
  selectedName: string | null
  selectedIndex: number
  onSelect: (name: string | null) => Promise<void>
  onHoverIndex: (index: number) => void
  handleKeyEvent: (key: KeyEvent) => boolean
}

export interface UseSkillsetSettingsOptions {
  onPublishSkillset?: (skillset: Skillset) => void
}

export function useSkillsetSettings(options: UseSkillsetSettingsOptions = {}): UseSkillsetSettingsResult {
  const { onPublishSkillset } = options
  const storage = useStorage()
  const [availableSkillsets, setAvailableSkillsets] = useState<SkillsetInfo[]>([])
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(0)

  useEffect(() => {
    // Load available skillsets
    Effect.runPromise(
      Effect.flatMap(SkillsetResolver, s => s.list()).pipe(
        Effect.provide(SkillsetResolverLive),
      )
    ).then((infos: SkillsetInfo[]) => {
      setAvailableSkillsets(infos)
    }).catch(() => {
      setAvailableSkillsets([])
    })

    // Load current selected skillset from config
    storage.config.loadFull().then(config => {
      setSelectedName(config.selectedSkillset ?? null)
    }).catch(() => {
      setSelectedName(null)
    })
  }, [storage])

  const onSelect = useCallback(async (name: string | null) => {
    await storage.config.updateFull(config => ({
      ...config,
      selectedSkillset: name ?? undefined,
    }))
    setSelectedName(name)

    if (onPublishSkillset) {
      if (!name) {
        onPublishSkillset(EmptySkillset)
      } else {
        Effect.runPromise(
          Effect.flatMap(SkillsetResolver, s => s.resolve(name)).pipe(
            Effect.provide(SkillsetResolverLive),
            Effect.catchAll(() => Effect.succeed(EmptySkillset)),
          )
        ).then(skillset => {
          onPublishSkillset(skillset)
        }).catch(() => {
          onPublishSkillset(EmptySkillset)
        })
      }
    }
  }, [storage, onPublishSkillset])

  const onHoverIndex = useCallback((index: number) => {
    setSelectedIndex(index)
  }, [])

  const handleKeyEvent = useCallback((key: KeyEvent): boolean => {
    const plain = !key.ctrl && !key.meta && !key.option
    if (!plain) return false

    if (key.name === 'up') {
      setSelectedIndex(i => Math.max(0, i - 1))
      return true
    }
    if (key.name === 'down') {
      setSelectedIndex(i => Math.min(availableSkillsets.length - 1, i + 1))
      return true
    }
    if (key.name === 'return' || key.name === 'enter') {
      const item = availableSkillsets[selectedIndex]
      if (item) {
        onSelect(item.name)
      }
      return true
    }
    return false
  }, [availableSkillsets, selectedIndex, onSelect])

  return {
    availableSkillsets,
    selectedName,
    selectedIndex,
    onSelect,
    onHoverIndex,
    handleKeyEvent,
  }
}
