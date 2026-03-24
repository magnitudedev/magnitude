import type { KeyEvent } from '@opentui/core'
import { PROVIDERS, createCodingAgentClient, type AgentStatusState, type ModelSelection, type ProviderAuthMethodStatus, type ProviderDefinition, type DetectedProvider } from '@magnitudedev/agent'
import type { MagnitudeSlot } from '@magnitudedev/agent'
import type { SettingsTab } from '../hooks/use-settings-navigation'
import type { WizardStep } from './setup-wizard-overlay'
import { BrowserSetupOverlay } from './browser-setup-overlay'
import { SetupWizardOverlay } from './setup-wizard-overlay'
import { RecentChatsOverlay } from './recent-chats-overlay'
import { ForkDetailOverlay } from './fork-detail-overlay'
import { SettingsOverlay } from './settings-overlay'
import { AuthMethodOverlay } from './auth-method-overlay'
import { LocalProviderOverlay } from './local-provider-overlay'
import { ApiKeyOverlay } from './api-key-overlay'
import { OAuthOverlay } from './oauth-overlay'
import type { RecentChat } from '../data/recent-chats'
import type { useAuthFlow } from '../hooks/use-auth-flow'

type AgentClient = Awaited<ReturnType<typeof createCodingAgentClient>>

export type AppOverlaysProps = {
  showSetupWizard: boolean
  wizardStep: WizardStep
  wizardTotalSteps: number
  wizardSlotModels: Record<MagnitudeSlot, ModelSelection | null>
  wizardConnectedProvider: string | null
  wizardProviderSelectedIndex: number
  wizardModelSelectedIndex: number
  showBrowserSetup: boolean
  setShowBrowserSetup: (v: boolean) => void
  handleWizardBrowserComplete: () => void
  handleWizardProviderSelected: (...args: any[]) => void
  handleWizardComplete: (...args: any[]) => void
  handleWizardBack: () => void
  handleWizardSkip: () => void
  setWizardProviderSelectedIndex: (n: number) => void
  setWizardModelSelectedIndex: (n: number) => void
  wizardProviders: ProviderDefinition[]
  onWizardCtrlCExit: () => void

  authFlow: ReturnType<typeof useAuthFlow>
  authMethodSelectedIndex: number
  setAuthMethodSelectedIndex: (n: number) => void

  detectedProviders: DetectedProvider[]
  connectedProviders: ProviderDefinition[]
  slotModels: Record<MagnitudeSlot, ModelSelection | null>
  selectingModelFor: MagnitudeSlot | null
  setSelectingModelFor: (v: MagnitudeSlot | null) => void
  preferencesSelectedIndex: number
  setPreferencesSelectedIndex: (n: number) => void
  providerDetailStatus: ProviderAuthMethodStatus | null
  providerDetailActions: Array<{ type: 'connect' | 'disconnect' | 'update-key'; methodIndex: number; label: string }>
  providerDetailSelectedIndex: number
  setProviderDetailSelectedIndex: (n: number) => void
  settingsTab: SettingsTab | null
  handleSettingsTabChange: (tab: SettingsTab) => void
  handleModelSelect: (providerId: string, modelId: string) => void | Promise<void>
  modelSearch: string
  onModelSearchChange: (value: string) => void
  showAllProviders: boolean
  onToggleShowAllProviders: () => void
  showRecommendedOnly: boolean
  onToggleShowRecommendedOnly: () => void
  handleProviderSelect: (providerId: string) => void
  handleProviderDetailAction: (idx: number) => void
  handleProviderDetailBack: () => void
  onBackFromModelPicker: () => void
  handleChangeSlot: (slot: MagnitudeSlot) => void
  modelTabHandleKeyEvent: (key: KeyEvent) => boolean
  providerTabHandleKeyEvent: (key: KeyEvent) => boolean
  modelNavigation: { items: any[]; selectedIndex: number; setSelectedIndex: (n: number) => void }
  providerNavigation: { selectedIndex: number; setSelectedIndex: (n: number) => void }
  onSettingsClose: () => void

  showRecentChatsOverlay: boolean
  recentChats: RecentChat[] | null
  recentChatsSelectedIndex: number
  setRecentChatsSelectedIndex: (n: number) => void
  setShowRecentChatsOverlay: (v: boolean | ((prev: boolean) => boolean)) => void
  handleResumeChat: (chat: RecentChat) => void

  expandedForkId: string | null
  client: AgentClient | null
  agentStatusState: AgentStatusState | null
  popForkOverlay: () => void
  pushForkOverlay: (forkId: string) => void
  onFileClick?: (path: string, section?: string) => void

  localProviderConfig: { baseUrl?: string; modelId?: string } | null
}

export function AppOverlays({
  showSetupWizard,
  wizardStep,
  wizardTotalSteps,
  wizardSlotModels,
  wizardConnectedProvider,
  wizardProviderSelectedIndex,
  wizardModelSelectedIndex,
  showBrowserSetup,
  setShowBrowserSetup,
  handleWizardBrowserComplete,
  handleWizardProviderSelected,
  handleWizardComplete,
  handleWizardBack,
  handleWizardSkip,
  setWizardProviderSelectedIndex,
  setWizardModelSelectedIndex,
  wizardProviders,
  onWizardCtrlCExit,
  authFlow,
  authMethodSelectedIndex,
  setAuthMethodSelectedIndex,
  detectedProviders,
  connectedProviders,
  slotModels,
  selectingModelFor,
  setSelectingModelFor,
  preferencesSelectedIndex,
  setPreferencesSelectedIndex,
  providerDetailStatus,
  providerDetailActions,
  providerDetailSelectedIndex,
  setProviderDetailSelectedIndex,
  settingsTab,
  handleSettingsTabChange,
  handleModelSelect,
  modelSearch,
  onModelSearchChange,
  showAllProviders,
  onToggleShowAllProviders,
  showRecommendedOnly,
  onToggleShowRecommendedOnly,
  handleProviderSelect,
  handleProviderDetailAction,
  handleProviderDetailBack,
  onBackFromModelPicker,
  handleChangeSlot,
  modelTabHandleKeyEvent,
  providerTabHandleKeyEvent,
  modelNavigation,
  providerNavigation,
  onSettingsClose,
  showRecentChatsOverlay,
  recentChats,
  recentChatsSelectedIndex,
  setRecentChatsSelectedIndex,
  setShowRecentChatsOverlay,
  handleResumeChat,
  expandedForkId,
  client,
  agentStatusState,
  popForkOverlay,
  pushForkOverlay,
  onFileClick,
  localProviderConfig,
}: AppOverlaysProps) {
  if (showSetupWizard && wizardStep === 'browser') {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <BrowserSetupOverlay
          onClose={() => handleWizardBrowserComplete()}
          onResult={() => handleWizardBrowserComplete()}
          wizardMode={{
            stepLabel: `Browser (${wizardTotalSteps} of ${wizardTotalSteps})`,
            subtitle: 'The browser agent requires Chromium to control web pages.',
            onSkip: handleWizardSkip,
            onBack: handleWizardBack,
          }}
        />
      </box>
    )
  }

  if (showSetupWizard && !authFlow.oauthState && !authFlow.apiKeySetup && !authFlow.showLocalSetup && !authFlow.showAuthMethodOverlay) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <SetupWizardOverlay
          step={wizardStep}
          allProviders={wizardProviders}
          detectedProviders={detectedProviders}
          slotModels={wizardSlotModels}
          connectedProviderName={wizardConnectedProvider}
          totalSteps={wizardTotalSteps}
          onProviderSelected={handleWizardProviderSelected}
          onComplete={handleWizardComplete}
          onBack={handleWizardBack}
          onSkip={handleWizardSkip}
          onWizardCtrlCExit={onWizardCtrlCExit}
          providerSelectedIndex={wizardProviderSelectedIndex}
          onProviderSelectedIndexChange={setWizardProviderSelectedIndex}
          onProviderHoverIndex={setWizardProviderSelectedIndex}
          modelNavSelectedIndex={wizardModelSelectedIndex}
          onModelNavSelectedIndexChange={setWizardModelSelectedIndex}
          onModelNavHoverIndex={setWizardModelSelectedIndex}
        />
      </box>
    )
  }

  if (showRecentChatsOverlay) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <RecentChatsOverlay
          chats={recentChats ?? []}
          selectedIndex={recentChatsSelectedIndex}
          onSelectedIndexChange={setRecentChatsSelectedIndex}
          onSelect={handleResumeChat}
          onHoverIndex={setRecentChatsSelectedIndex}
          onClose={() => setShowRecentChatsOverlay(false)}
        />
      </box>
    )
  }

  if (expandedForkId && client) {
    const agentId = agentStatusState?.agentByForkId.get(expandedForkId)
    const agent = agentId ? agentStatusState?.agents.get(agentId) : undefined
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <ForkDetailOverlay
          forkId={expandedForkId}
          forkName={agent?.name ?? 'Agent'}
          forkRole={agent?.role ?? 'agent'}
          onClose={popForkOverlay}
          onForkExpand={pushForkOverlay}
          onFileClick={onFileClick}
          subscribeForkDisplay={(fId, cb) => client.state.display.subscribeFork(fId, cb)}
        />
      </box>
    )
  }

  if (showBrowserSetup) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <BrowserSetupOverlay
          onClose={() => setShowBrowserSetup(false)}
          onResult={() => setShowBrowserSetup(false)}
        />
      </box>
    )
  }

  if (settingsTab !== null) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <SettingsOverlay
          activeTab={settingsTab}
          onTabChange={handleSettingsTabChange}
          onClose={onSettingsClose}
          modelItems={modelNavigation.items}
          modelSelectedIndex={modelNavigation.selectedIndex}
          onModelSelect={handleModelSelect}
          onModelHoverIndex={modelNavigation.setSelectedIndex}
          modelSearch={modelSearch}
          onModelSearchChange={onModelSearchChange}
          showAllProviders={showAllProviders}
          onToggleShowAllProviders={onToggleShowAllProviders}
          showRecommendedOnly={showRecommendedOnly}
          onToggleShowRecommendedOnly={onToggleShowRecommendedOnly}
          allProviders={PROVIDERS}
          detectedProviders={detectedProviders}
          providerSelectedIndex={providerNavigation.selectedIndex}
          onProviderSelect={handleProviderSelect}
          onProviderHoverIndex={providerNavigation.setSelectedIndex}
          providerDetailStatus={providerDetailStatus}
          providerDetailActions={providerDetailActions}
          providerDetailSelectedIndex={providerDetailSelectedIndex}
          onProviderDetailAction={handleProviderDetailAction}
          onProviderDetailHoverIndex={setProviderDetailSelectedIndex}
          slotModels={slotModels}
          selectingModelFor={selectingModelFor}
          onChangeSlot={handleChangeSlot}
          modelPrefsSelectedIndex={preferencesSelectedIndex}
          onModelPrefsHoverIndex={setPreferencesSelectedIndex}
          localProviderConfig={localProviderConfig}
          localProviderAuth={(() => {
            const localDetected = detectedProviders.find((d) => d.provider.id === 'local')
            return localDetected?.auth?.type === 'api' ? localDetected.auth : null
          })()}
          onModelHandleKeyEvent={modelTabHandleKeyEvent}
          onProviderHandleKeyEvent={providerTabHandleKeyEvent}
          onBackFromModelPicker={onBackFromModelPicker}
          onBackFromProviderDetail={handleProviderDetailBack}
        />
      </box>
    )
  }

  if (authFlow.showAuthMethodOverlay && authFlow.authMethodProvider) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <AuthMethodOverlay
          providerName={authFlow.authMethodProvider.name}
          methods={authFlow.authMethodProvider.authMethods}
          selectedIndex={authMethodSelectedIndex}
          onSelectedIndexChange={setAuthMethodSelectedIndex}
          onSelect={(methodIndex) => authFlow.startAuthForProvider(authFlow.authMethodProvider!, methodIndex)}
          onHoverIndex={setAuthMethodSelectedIndex}
          onBack={authFlow.closeAuthMethodPicker}
          wizardMode={showSetupWizard ? {
            stepLabel: `Providers (1 of ${wizardTotalSteps})`,
            subtitle: `Connect to ${authFlow.authMethodProvider.name} to get started.`,
            onSkip: handleWizardSkip,
            onBack: authFlow.closeAuthMethodPicker,
          } : undefined}
        />
      </box>
    )
  }

  if (authFlow.showLocalSetup) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <LocalProviderOverlay
          initialConfig={{ url: localProviderConfig?.baseUrl ?? '', modelId: localProviderConfig?.modelId ?? '' }}
          onSubmit={authFlow.handleLocalSetupSubmit}
          onCancel={authFlow.handleLocalSetupCancel}
          wizardMode={showSetupWizard ? {
            stepLabel: `Providers (1 of ${wizardTotalSteps})`,
            subtitle: 'Configure your local provider to get started.',
            onSkip: handleWizardSkip,
            onBack: authFlow.handleLocalSetupCancel,
          } : undefined}
        />
      </box>
    )
  }

  if (authFlow.apiKeySetup) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <ApiKeyOverlay
          providerName={authFlow.apiKeySetup.provider.name}
          envKeyHint={authFlow.apiKeySetup.envKeyHint}
          initialKey={authFlow.apiKeySetup.existingKey}
          onSubmit={authFlow.handleApiKeySubmit}
          onCancel={authFlow.handleApiKeyCancel}
          wizardMode={showSetupWizard ? {
            stepLabel: `Providers (1 of ${wizardTotalSteps})`,
            subtitle: `Connect to ${authFlow.apiKeySetup.provider.name} to get started.`,
            onSkip: handleWizardSkip,
            onBack: authFlow.handleApiKeyCancel,
          } : undefined}
        />
      </box>
    )
  }

  if (authFlow.oauthState) {
    return (
      <box style={{ flexDirection: 'column', height: '100%' }}>
        <OAuthOverlay
          providerName={authFlow.oauthState.provider.name}
          mode={authFlow.oauthState.mode}
          url={authFlow.oauthState.url}
          onSubmitCode={authFlow.handleOAuthCodeSubmit}
          codeError={authFlow.oauthState.codeError}
          isSubmitting={authFlow.oauthState.isSubmitting}
          userCode={authFlow.oauthState.userCode}
          onCancel={authFlow.handleOAuthCancel}
          onCopyUrl={authFlow.handleOAuthCopyUrl}
          onCopyCode={authFlow.handleOAuthCopyCode}
          wizardMode={showSetupWizard ? {
            stepLabel: `Providers (1 of ${wizardTotalSteps})`,
            subtitle: `Connect to ${authFlow.oauthState.provider.name} to get started.`,
            onSkip: handleWizardSkip,
            onBack: authFlow.handleOAuthCancel,
          } : undefined}
        />
      </box>
    )
  }

  return null
}