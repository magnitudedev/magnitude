import { expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

const captured: Array<Record<string, unknown>> = []

mock.module('./fork-detail-overlay', () => ({
  ForkDetailOverlay: (props: Record<string, unknown>) => {
    captured.push(props)
    return <text>[fork-overlay]</text>
  },
}))

const { AppOverlays } = await import('./app-overlays')

test('passes fork-local file viewer dependencies into fork overlay without root file click prop', () => {
  captured.length = 0

  renderToStaticMarkup(
    <AppOverlays
      showSetupWizard={false}
      wizardStep={'provider' as any}
      wizardTotalSteps={1}
      wizardHasProviderEndpointStep={false}
      wizardSlotModels={{} as any}
      wizardConnectedProvider={null}
      wizardSelectedProviderId={null}
      wizardSelectedProviderDiscoveredModels={[]}
      wizardSelectedProviderRememberedModelIds={[]}
      wizardProviderSelectedIndex={0}
      wizardModelSelectedIndex={0}
      showBrowserSetup={false}
      setShowBrowserSetup={() => {}}
      handleWizardBrowserComplete={() => {}}
      handleWizardProviderSelected={() => {}}
      handleWizardComplete={() => {}}
      handleWizardContinueFromLocalProvider={() => {}}
      handleWizardBack={() => {}}
      handleWizardSkip={() => {}}
      onLocalProviderSaveEndpoint={() => {}}
      onLocalProviderRefreshModels={() => {}}
      onLocalProviderAddManualModel={() => {}}
      onLocalProviderRemoveManualModel={() => {}}
      onLocalProviderSaveOptionalApiKey={() => {}}
      setWizardProviderSelectedIndex={() => {}}
      setWizardModelSelectedIndex={() => {}}
      wizardProviders={[]}
      onWizardCtrlCExit={() => {}}
      authFlow={{} as any}
      authMethodSelectedIndex={0}
      setAuthMethodSelectedIndex={() => {}}
      detectedProviders={[]}
      connectedProviders={[]}
      slotModels={{} as any}
      selectingModelFor={null}
      setSelectingModelFor={() => {}}
      preferencesSelectedIndex={0}
      setPreferencesSelectedIndex={() => {}}
      providerDetailStatus={null}
      providerDetailOptions={undefined}
      providerDetailActions={[]}
      providerDetailSelectedIndex={0}
      setProviderDetailSelectedIndex={() => {}}
      settingsTab={null}
      handleSettingsTabChange={() => {}}
      handleModelSelect={() => {}}
      modelSearch=""
      onModelSearchChange={() => {}}
      showAllProviders={false}
      onToggleShowAllProviders={() => {}}
      showRecommendedOnly={false}
      onToggleShowRecommendedOnly={() => {}}
      handleProviderSelect={() => {}}
      handleProviderDetailAction={() => {}}
      handleProviderDetailBack={() => {}}
      onBackFromModelPicker={() => {}}
      presets={[]}
      systemDefaultsPresetToken=""
      onSavePreset={() => {}}
      onLoadPreset={() => {}}
      onDeletePreset={() => {}}
      handleChangeSlot={() => {}}
      modelTabHandleKeyEvent={() => false}
      providerTabHandleKeyEvent={() => false}
      modelNavigation={{ items: [], selectedIndex: 0, setSelectedIndex: () => {} }}
      providerNavigation={{ selectedIndex: 0, setSelectedIndex: () => {} }}
      onSettingsClose={() => {}}
      showRecentChatsOverlay={false}
      recentChats={[]}
      recentChatsSelectedIndex={0}
      setRecentChatsSelectedIndex={() => {}}
      setShowRecentChatsOverlay={() => {}}
      handleResumeChat={() => {}}
      expandedForkId="fork-1"
      client={{
        state: {
          display: { subscribeFork: () => () => {} },
          compaction: { subscribeFork: () => () => {} },
        },
      } as any}
      agentStatusState={null}
      forkModelSummary={null}
      forkContextHardCap={null}
      popForkOverlay={() => {}}
      pushForkOverlay={() => {}}
      workspacePath="/tmp/workspace"
      projectRoot="/tmp/project"
    />,
  )

  expect(captured.length).toBe(1)
  expect(captured[0]?.workspacePath).toBe('/tmp/workspace')
  expect(captured[0]?.projectRoot).toBe('/tmp/project')
  expect('onFileClick' in (captured[0] ?? {})).toBe(false)
})