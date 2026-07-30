import { TextAttributes } from "@opentui/core";
import { useKeyboard } from "@opentui/react";
import { Option } from "effect";
import type {
  AcnInstallationPhase,
  AcnLifecycleState,
} from "@magnitudedev/sdk";
import { ProgressBar } from "../../components/progress-bar";
import { useTheme } from "../../hooks/use-theme";

const PHASE_LABELS: Readonly<Record<AcnInstallationPhase, string>> = {
  DownloadingDaemon: "Downloading daemon",
  DownloadingInferenceEngine: "Downloading inference engine",
  StartingMagnitude: "Starting Magnitude",
};

const INSTALLATION_PANEL_WIDTH = 64;
const PROGRESS_BAR_WIDTH = 36;
const PERCENTAGE_WIDTH = 5;

const formatBytes = (bytes: number): string => {
  const mebibytes = bytes / (1024 * 1024);
  return `${mebibytes.toFixed(1)} MiB`;
};

export function AcnBootstrapScreen({
  state,
  onRetry,
  onQuit,
}: {
  readonly state: AcnLifecycleState;
  readonly onRetry: () => void;
  readonly onQuit: () => void;
}) {
  const theme = useTheme();

  useKeyboard((key) => {
    const name = key.name.toLowerCase();
    const isCtrlC =
      key.ctrl && name === "c" && !key.meta && !key.option;
    if (isCtrlC) {
      key.preventDefault();
      key.stopPropagation();
      onQuit();
      return;
    }
    if (state._tag !== "Failed" || (name !== "r" && name !== "q")) return;

    key.preventDefault();
    key.stopPropagation();
    if (name === "r") {
      onRetry();
    } else {
      onQuit();
    }
  });

  if (state._tag === "Ready" || state._tag === "Checking") return null;

  if (state._tag === "Failed") {
    const title =
      state.stage === "AcquireAcn" ||
      state.stage === "PrepareLocalInference"
        ? "Magnitude failed to install"
        : "Magnitude failed to start";
    return (
      <box
        style={{
          backgroundColor: theme.background,
          flexGrow: 1,
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <text style={{ fg: theme.error }} attributes={TextAttributes.BOLD}>
          {title}
        </text>
        <box
          style={{
            marginTop: 1,
            maxWidth: 80,
            paddingLeft: 2,
            paddingRight: 2,
          }}
        >
          <text style={{ fg: theme.foreground }}>{state.message}</text>
        </box>
        <box style={{ marginTop: 2 }}>
          <text style={{ fg: theme.muted }}>R Retry Q Quit</text>
        </box>
      </box>
    );
  }

  const installing = state._tag === "Installing";
  const percentage = installing ? Math.floor(state.overallProgress * 100) : 0;
  const downloadDetail =
    installing &&
    state.phase !== "StartingMagnitude" &&
    state.detailIsExact &&
    Option.isSome(state.detail) &&
    state.detail.value.unit === "Bytes"
      ? ` · ${formatBytes(state.detail.value.completed)} of ${formatBytes(
          state.detail.value.totalBytes
        )}`
      : "";

  return (
    <box
      style={{
        backgroundColor: theme.background,
        flexGrow: 1,
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      {installing ? (
        <box
          style={{
            width: INSTALLATION_PANEL_WIDTH,
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <text
            style={{ fg: theme.foreground }}
            attributes={TextAttributes.BOLD}
          >
            Installing Magnitude
          </text>
          <box
            style={{
              marginTop: 1,
              flexDirection: "row",
              alignItems: "center",
            }}
          >
            <ProgressBar
              value={state.overallProgress}
              width={PROGRESS_BAR_WIDTH}
            />
            <box
              style={{
                width: PERCENTAGE_WIDTH,
                alignItems: "flex-end",
              }}
            >
              <text style={{ fg: theme.muted }}>{`${percentage}%`}</text>
            </box>
          </box>
          <box
            style={{
              marginTop: 1,
              width: INSTALLATION_PANEL_WIDTH,
              alignItems: "center",
            }}
          >
            <text style={{ fg: theme.muted }}>
              {`${PHASE_LABELS[state.phase]}${downloadDetail}`}
            </text>
          </box>
        </box>
      ) : (
        <box
          style={{
            width: INSTALLATION_PANEL_WIDTH,
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <text
            style={{ fg: theme.foreground }}
            attributes={TextAttributes.BOLD}
          >
            Starting Magnitude
          </text>
          <text style={{ fg: theme.muted }}>Starting local inference</text>
        </box>
      )}
    </box>
  );
}
