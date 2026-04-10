import { expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    muted: '#888888',
    primary: '#5e81ac',
    foreground: '#ffffff',
    border: '#4c566a',
    success: '#00ff00',
    surface: '#111111',
  }),
}))

const { AppOverlays } = await import('./app-overlays')

function renderForkOverlay(showCopiedToast: boolean) {
  return renderToStaticMarkup(
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
      showCopiedToast={showCopiedToast}
    />,
  )
}

test('renders fork overlay branch', () => {
  const html = renderForkOverlay(false)
  expect(html).toContain('Agent')
})

test('shows copy toast in fork overlay when clipboard toast state is active', () => {
  const html = renderForkOverlay(true)
  expect(html).toContain('Copied to clipboard')
})
