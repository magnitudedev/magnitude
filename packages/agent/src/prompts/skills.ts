import type { WorkflowSkill, Phase } from '@magnitudedev/skills'

export function formatPhasePrompt(phase: Phase, phaseIndex: number): string {
  const sections: string[] = []

  // Phase header and instructions
  const header = `## Phase ${phaseIndex + 1}: ${phase.name}`
  if (phase.prompt.trim()) {
    sections.push(`${header}\n\n${phase.prompt.trim()}`)
  } else {
    sections.push(header)
  }

  // Required submissions
  if (phase.submit && phase.submit.fields.length > 0) {
    const fieldLines = phase.submit.fields.map(f => {
      if (f.type === 'file') {
        const fileType = f.fileType ? ` (${f.fileType})` : ''
        return `- **${f.name}**${fileType}: ${f.description}`
      }
      return `- **${f.name}** (text): ${f.description}`
    })
    sections.push(`### Required Submissions\n\n${fieldLines.join('\n')}`)
  }

  return sections.join('\n\n')
}

export function formatSkillInitialPrompt(skill: WorkflowSkill): string {
  const sections: string[] = []

  // Always include preamble if present
  if (skill.preamble.trim()) {
    sections.push(skill.preamble.trim())
  }

  if (skill.phases.length > 0) {
    // Phase overview
    const phaseList = skill.phases.map((p, i) => `${i + 1}. **${p.name}**`).join('\n')
    sections.push(`## Workflow Phases\n\nThis skill has ${skill.phases.length} phase${skill.phases.length > 1 ? 's' : ''}:\n\n${phaseList}\n\nComplete each phase by submitting deliverables with \`phase-submit\`. Phases may have criteria that must pass before advancing.`)

    // First phase instructions with submission fields
    sections.push(formatPhasePrompt(skill.phases[0], 0))
  }

  return sections.join('\n\n')
}
