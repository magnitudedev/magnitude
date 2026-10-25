import { useMemo } from "react"
import { Atom, Result, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Option } from "effect"
import { CircleNotchIcon, XIcon } from "@phosphor-icons/react"
import {
  activeLocalModel, formatLocalModelDisplayName, formatModelLoadPercentage,
  isMeasuredModelLoadStage, useLocalModels, useLocalModelMutations, useLocalModelStopStatus,
} from "@magnitudedev/client-common"
import type { ModelInstanceAllocation, ModelLoadStage } from "@magnitudedev/sdk"
import { useSession } from "../session"
import { useReconnectAttempt, type ServiceObservation } from "../service-view"
import { Button } from "./ui/button"
import { ErrorNotice } from "./error-notice"
import { ModelLogo } from "./model-logo"
import { SkeletonLine } from "./page-skeletons"
import { MemoryFigures } from "./memory-breakdown"

const loadLabels: Record<ModelLoadStage, string> = {
  queued: "Waiting for memory",
  preparing: "Preparing model",
  optimizing: "Optimizing model",
  loading_weights: "Loading weights",
  finalizing: "Finalizing model",
}

export function SettingsStatus({ observation }: { observation: ServiceObservation }) {
  const session = useSession()
  const models = useLocalModels()
  const { stop } = useLocalModelMutations()
  const stopping = useLocalModelStopStatus()
  const retry = useAtomSet(session.retryService)
  const retrying = useAtomValue(session.retryService)
  const attempt = useReconnectAttempt()
  const reconnectAction = useMemo(() => Atom.fn((_: void) => attempt), [attempt])
  const reconnect = useAtomSet(reconnectAction)
  const reconnecting = useAtomValue(reconnectAction)
  const service = Result.isSuccess(observation.service) ? observation.service.value : null
  const ready = service?._tag === "Ready"
  const unavailable = service?._tag === "Failed" || service?._tag === "CleanupFailed" || service?._tag === "Unreachable" || Result.isFailure(observation.service)
  const reading = (!service && !unavailable) || (ready && Result.isInitial(models))
  const active = ready && Result.isSuccess(models) ? Option.getOrUndefined(activeLocalModel(models.value)) : undefined
  const residency = active?.residency
  const loading = residency?._tag === "Requested" || residency?._tag === "Loading"
  const activity = residency?._tag === "Requested" ? "Preparing model"
    : residency?._tag === "Loading" ? loadLabels[residency.stage]
    : residency?._tag === "Stopping" ? "Stopping model" : null
  const fraction = residency?._tag === "Loading" && isMeasuredModelLoadStage(residency.stage) ? residency.fraction : null
  const serviceLabel = ready ? "Magnitude service running" : unavailable ? "Magnitude service unavailable"
    : service?._tag === "Reconnecting" ? "Reconnecting to Magnitude" : "Magnitude service starting"
  const title = active ? formatLocalModelDisplayName(active.model)
    : unavailable ? "Magnitude is unavailable"
    : service?._tag === "Reconnecting" ? "Reconnecting to Magnitude"
    : !ready ? "Starting Magnitude"
    : Result.isFailure(models) ? "Model status unavailable"
    : "No model loaded"
  const allocation: Option.Option<ModelInstanceAllocation> | null = !ready || !Result.isSuccess(models) ? null
    : residency?._tag === "Ready" ? Option.some(residency.allocation)
    : residency?._tag === "Stopping" && residency.allocation._tag === "Resident" ? Option.some(residency.allocation.allocation)
    : residency ? null : Option.none()
  const pendingLabel = unavailable || Result.isFailure(models) ? "Unavailable" : loading ? "Available once loaded"
    : residency?._tag === "Stopping" ? "Releasing memory" : "Waiting for service"
  return <section aria-label="Status" className="mt-7">
    <h2 className="mb-2 px-1 text-xs font-medium uppercase tracking-wide text-slate-500">Status</h2>
    <div className="settings-status-card" aria-busy={reading}>
      <div className="settings-status-top">
        <div className="settings-status-model">
          {reading ? <SkeletonLine className="h-5 w-52" /> : <span className="settings-status-name" title={title}>
            {active && <ModelLogo model={active.model} className="settings-status-logo" />}{title}
          </span>}
          <span role="img" aria-label={serviceLabel} title={serviceLabel} className={`settings-service-indicator ${unavailable ? "is-failed" : ready ? "is-ready" : "is-starting"}`}>
            {unavailable ? <XIcon weight="bold" /> : ready ? <span /> : <CircleNotchIcon className="motion-safe:animate-spin" />}
          </span>
        </div>
        <div className="settings-status-controls">
          {activity && <span className="settings-status-activity" role="status"><CircleNotchIcon aria-hidden="true" className="motion-safe:animate-spin" /><span>{activity}</span>{fraction !== null && <strong>{formatModelLoadPercentage(fraction)}</strong>}</span>}
          {active && <Button size="sm" variant="outline" className="settings-status-action" disabled={stopping.pending || residency?._tag === "Stopping"} onClick={() => stop()}>{loading ? "Cancel" : "Stop model"}</Button>}
          {service?._tag === "Failed" && observation.canRetry && <Button size="sm" variant="outline" disabled={retrying.waiting} onClick={() => retry()}>Retry service</Button>}
          {(service?._tag === "Unreachable" || service?._tag === "Reconnecting") && <Button size="sm" variant="outline" disabled={reconnecting.waiting} onClick={() => reconnect()}>Reconnect</Button>}
        </div>
      </div>
      <MemoryFigures allocation={allocation} reading={reading} pendingLabel={pendingLabel} />
      <div className="empty:hidden space-y-2 px-4 pb-3">
        {service && "message" in service && <ErrorNotice title={service._tag === "CleanupFailed" ? "Couldn’t confirm the service has stopped" : "The service couldn’t start"} description={service._tag === "CleanupFailed" ? "Some background work may still be running. Quit Magnitude to retry cleanup." : service.message} />}
        {Result.isFailure(observation.service) && <ErrorNotice title="Couldn’t read service status" description="Magnitude can’t confirm the service’s current state." />}
        {service?._tag === "Unreachable" && <ErrorNotice title="Can’t reach Magnitude" description="Check that Magnitude is running and this device can reach it. This page keeps trying to reconnect." />}
        {Result.isFailure(models) && ready && <ErrorNotice title="Couldn’t read model status" description="Magnitude can’t confirm which model is loaded or its memory usage." />}
        {Option.isSome(stopping.failure) && <ErrorNotice title="Couldn’t stop the model" description={stopping.failure.value} />}
        {Result.isFailure(retrying) && !retrying.waiting && <ErrorNotice title="Couldn’t retry the service" description="Try again, or restart Magnitude." />}
        {Result.isFailure(reconnecting) && !reconnecting.waiting && <ErrorNotice title="Couldn’t reconnect" description="Check that Magnitude is running. This page keeps trying automatically." />}
        {Option.isSome(observation.tray) && observation.tray.value._tag === "Unavailable" && <ErrorNotice severity="warning" title="The tray icon isn’t available" description="Closing this window keeps Magnitude running. Open it again from your applications menu." />}
      </div>
    </div>
  </section>
}
