import { renderToStaticMarkup } from "react-dom/server"
import { expect, test, vi } from "vitest"
import type { LocalInferenceOnboardingSnapshot } from "@magnitudedev/sdk"

vi.mock("@opentui/react", () => ({
  useKeyboard: () => {},
  useRenderer: () => ({ setMousePointer: () => {}, requestRender: () => {} }),
}))
vi.mock("@magnitudedev/client-common", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@magnitudedev/client-common")>()),
  LOGO_LINES: ["M"],
}))
vi.mock("../app-shell/login", () => ({ MagnitudeLoginScreen: () => null }))
vi.mock("../../hooks/use-theme", () => ({
  useTheme: () => ({
    primary: "blue",
    foreground: "white",
    muted: "gray",
    border: "gray",
    warning: "yellow",
    error: "red",
  }),
}))
vi.mock("../../components/button", () => ({
  Button: ({ children }: { children?: unknown }) => <button>{children as never}</button>,
}))
const {
  LOCAL_MODEL_SECTION_WIDTH,
  LocalUsageSetupView,
  ModelSetupOnboardingView,
  LocalInferenceOnboardingView,
  localModelSectionRule,
  moveLocalUsageFocus,
} = await import("./screen")

const controller = {
  operationId: null,
  downloadConfigurationId: null,
  progress: null,
  error: null,
  busy: false,
  usageSnapshot: null,
  configureUsage: async () => snapshot,
  startDownload: async () => {},
  cancelDownload: async () => {},
  activate: async () => true,
  configureCloud: async () => {},
  completeOnboarding: async () => true,
  cloudKeyAlreadySet: false,
} as const

const snapshot: LocalInferenceOnboardingSnapshot = {
  onboarding: { required: true },
  configuration: { usable: false },
  usage: {
    selection: { localModelRole: "main", sessionConcurrency: "one" },
  },
  runtime: {
    status: "ready",
    canDownload: true,
    canActivate: true,
  },
  capabilities: {
    binary: { identity: "managed" },
    system: { totalMemoryBytes: 32 * 1024 ** 3 },
    accelerators: [{
      id: "MTL0",
      backend: "Metal",
      description: "Apple GPU",
      capacityBytes: 28 * 1024 ** 3,
      capacityKind: "recommended-working-set",
      memoryDomainId: "unified:0",
      sharesSystemMemory: true,
    }],
    warnings: [],
  },
  running: [],
  downloaded: [],
  recommendations: [{
    configurationId: "qwen@ctx-32768",
    catalogModelId: "qwen",
    badge: "recommended",
    displayName: "Qwen3.6 35B-A3B",
    family: "qwen3.6",
    architecture: "moe",
    totalParametersBillions: 35,
    activeParametersBillions: 3,
    quantization: {
      format: "UD-Q4_K_XL",
      bitsClass: "q4",
      quantAwareCheckpoint: false,
      fidelityLabel: "Good fidelity with some possible quality loss",
      fidelityEvidence: "Model-specific low-divergence evidence; this is not coding accuracy.",
      fidelitySourceUrl: "https://arxiv.org/abs/2606.19558",
    },
    repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
    revision: "a483e9e6cbd595906af30beda3187c2663a1118c",
    quantTag: "UD-Q4_K_XL",
    files: [{
      path: "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
      sizeBytes: 22_360_456_160,
      sha256: "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450",
      downloadUrl: "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    }],
    totalDownloadBytes: 22_360_456_160,
    sourcePageUrl: "https://huggingface.co/source",
    license: { id: "apache-2.0", url: "https://www.apache.org/licenses/LICENSE-2.0", acknowledgementRequired: false },
    contextTokens: 200_000,
    servingProfile: {
      localModelRole: "main",
      sessionConcurrency: "one",
      parallelSlots: 1,
      contextTokensPerSlot: 200_000,
      totalContextCapacityTokens: 200_000,
      slotAllocation: "uniform",
      runtimeProfileId: "test-profile",
    },
    modelMaximumContextTokens: 262_144,
    estimatedRuntimeBytes: 25_000_000_000,
    stableCapacityBudgetBytes: 25_769_803_776,
    fitMarginBytes: 769_803_776,
    fitClass: "cpu_or_unified",
    constrainedContext: false,
    explanation: "uses total capacity",
  }],
  warnings: [],
}

test("asks both usage questions before displaying dynamic recommendations", () => {
  const html = renderToStaticMarkup(
    <ModelSetupOnboardingView
      snapshot={snapshot}
      onExit={() => {}}
      onComplete={() => {}}
      controller={controller}
    />,
  )
  expect(html).toContain("How do you plan to use local models?")
  expect(html).toContain("Magnitude uses llama.cpp to run local models in the background.")
  expect(html).toContain("Answer two questions and we&#x27;ll recommend Hugging Face models that fit your setup.")
  expect(html).toContain("As my main agent")
  expect(html).toContain("For local subagents")
  expect(html).toContain("Uses a cloud main agent and reserves three context windows for local subagents.")
  expect(html).toContain("How many Magnitude sessions will you run at once?")
  expect(html).toContain("One session")
  expect(html).toContain("Multiple sessions")
  expect(html).toContain("smaller recommended local model")
  expect(html).toContain("See recommendations")
  expect(html).not.toContain("Qwen3.6 35B-A3B")
})

test("renders a detected external llama.cpp server without a separate attachment choice", () => {
  const html = renderToStaticMarkup(
    <LocalUsageSetupView
      snapshot={{
        ...snapshot,
        running: [{
          choiceId: "running-qwen",
          source: "running",
          displayName: "Qwen3.6 35B-A3B",
          providerModelId: "qwen-running",
          quantization: {
            format: "Q6_K",
            bitsClass: "q6",
            quantAwareCheckpoint: false,
            fidelityLabel: "Very high fidelity with minimal quality loss",
            fidelityEvidence: "Reported by server",
            fidelitySourceUrl: "https://github.com/ggml-org/llama.cpp",
          },
          contextTokens: 200_000,
          fitClass: "unknown",
          managed: false,
          compatible: true,
          explanation: "running",
        }],
      }}
      localModelRole="main"
      sessionConcurrency="one"
      onSelectRole={() => {}}
      onSelectConcurrency={() => {}}
      onContinue={() => {}}
      onSkip={() => {}}
      onExit={() => {}}
      busy={false}
    />,
  )
  expect(html).toContain("llama.cpp server detected")
  expect(html).toContain("Qwen3.6 35B-A3B · Q6_K · 200K context · Running outside Magnitude")
  expect(html).not.toContain("Use Running Server")
  expect(html).not.toContain("we’ll check")
})

test("moves one keyboard focus cursor across all choices and continue", () => {
  expect(moveLocalUsageFocus(0, -1)).toBe(0)
  expect(moveLocalUsageFocus(0, 1)).toBe(1)
  expect(moveLocalUsageFocus(3, 1)).toBe(4)
  expect(moveLocalUsageFocus(4, 1)).toBe(4)
})

test("renders stable capacity and exact recommendation metadata without an API-key gate", () => {
  const html = renderToStaticMarkup(
    <LocalInferenceOnboardingView
      snapshot={snapshot}
      onExit={() => {}}
      onConfigured={() => {}}
      onSkip={() => {}}
      onBack={() => {}}
      controller={{
        operationId: null,
        downloadConfigurationId: null,
        progress: null,
        error: null,
        busy: false,
        usageSnapshot: null,
        configureUsage: async () => snapshot,
        startDownload: async () => {},
        cancelDownload: async () => {},
        activate: async () => true,
        configureCloud: async () => {},
        completeOnboarding: async () => true,
        cloudKeyAlreadySet: false,
      }}
    />,
  )
  expect(html).toContain("LOCAL MODEL SETUP")
  expect(html).toContain("32.0 GiB total system memory")
  expect(html).toContain("Metal: Apple GPU (28.0 GiB total)")
  expect(html).toContain("Qwen3.6 35B-A3B")
  expect(html).toContain("UD-Q4_K_XL")
  expect(html).toContain("200K context")
  expect(html).toContain("UD-Q4_K_XL · 22.4 GB · 35B total / 3B active · 200K context")
  expect(html).toContain("Good fidelity with some possible quality loss")
  expect(html).toContain("Skip for now (Esc)")
  expect(html).not.toContain("Recommendations use total capacity")
  expect(html).not.toContain("cloud fallback")
  expect(html).not.toContain("Paste your API key")
})

test("ends every model section heading at the same column", () => {
  for (const label of ["RUNNING NOW", "DOWNLOADED", "POSSIBLE DOWNLOADS", "RECOMMENDED DOWNLOADS"]) {
    expect(`${label}  ${localModelSectionRule(label)}`).toHaveLength(LOCAL_MODEL_SECTION_WIDTH)
  }
  expect(localModelSectionRule("RUNNING NOW").length).toBeGreaterThan(
    localModelSectionRule("POSSIBLE DOWNLOADS").length,
  )
})

test("shows recommendations while the CTO-owned llama.cpp bootstrap boundary is pending", () => {
  const html = renderToStaticMarkup(
    <LocalInferenceOnboardingView
      snapshot={{
        ...snapshot,
        runtime: {
          status: "integration_pending",
          canDownload: false,
          canActivate: false,
          diagnostic: "Waiting for managed llama.cpp binary detection and installation",
        },
      }}
      onExit={() => {}}
      onConfigured={() => {}}
      onSkip={() => {}}
      onBack={() => {}}
      controller={{
        operationId: null,
        downloadConfigurationId: null,
        progress: null,
        error: null,
        busy: false,
        usageSnapshot: null,
        configureUsage: async () => snapshot,
        startDownload: async () => {},
        cancelDownload: async () => {},
        activate: async () => true,
        configureCloud: async () => {},
        completeOnboarding: async () => true,
        cloudKeyAlreadySet: false,
      }}
    />,
  )
  expect(html).toContain("Install llama.cpp to download or run a recommended model")
  expect(html).toContain("Skip for now")
  expect(html).toContain("Qwen3.6 35B-A3B")
})

test("separates running inventory from possible downloads", () => {
  const html = renderToStaticMarkup(
    <LocalInferenceOnboardingView
      snapshot={{
        ...snapshot,
        running: [{
          choiceId: "running-qwen",
          source: "running",
          displayName: "Qwen3.6 35B-A3B",
          providerModelId: "qwen-running",
          quantization: {
            format: "Q6_K",
            bitsClass: "q6",
            quantAwareCheckpoint: false,
            fidelityLabel: "Server-reported Q6 fidelity",
            fidelityEvidence: "Reported by server",
            fidelitySourceUrl: "https://github.com/ggml-org/llama.cpp",
          },
          sizeBytes: 32_600_719_872,
          totalParametersBillions: 35.505251456,
          contextTokens: 200_192,
          fitClass: "unknown",
          managed: false,
          compatible: true,
          explanation: "running",
        }],
        recommendations: snapshot.recommendations.map((recommendation) => ({
          ...recommendation,
          quantization: {
            ...recommendation.quantization,
            format: "UD-Q8_K_XL",
            bitsClass: "q8" as const,
          },
          quantTag: "UD-Q8_K_XL",
        })),
      }}
      onExit={() => {}}
      onConfigured={() => {}}
      onSkip={() => {}}
      onBack={() => {}}
      controller={{
        operationId: null,
        downloadConfigurationId: null,
        progress: null,
        error: null,
        busy: false,
        usageSnapshot: null,
        configureUsage: async () => snapshot,
        startDownload: async () => {},
        cancelDownload: async () => {},
        activate: async () => true,
        configureCloud: async () => {},
        completeOnboarding: async () => true,
        cloudKeyAlreadySet: false,
      }}
    />,
  )
  expect(html).toContain("RUNNING NOW")
  expect(html).toContain("POSSIBLE DOWNLOADS")
  expect(html).toContain("Q6_K · 32.6 GB · 35.5B parameters · 200K context")
  expect(html).not.toContain("Server-reported Q6 fidelity")
  expect(html).toContain("Already Running")
  expect(html).toContain("Recommended")
  expect(html).not.toContain("cpu or unified")
  expect(html).not.toContain("unknown")
})
