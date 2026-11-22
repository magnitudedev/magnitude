import { Option } from "effect"
import type { LocalInferenceHardware, ModelInstanceAllocation, ModelStreamingStatus } from "@magnitudedev/sdk"
import {
  deriveHardwareMemoryView,
  formatMemoryCapacity,
  memoryDomainBarLayout,
  participatingMemoryDomains,
  streamingResidencySummary,
  type MemoryDomainBarLayout,
} from "@magnitudedev/client-common"
import { SkeletonLine } from "./page-skeletons"
import { Skeleton } from "./ui/skeleton"

function DomainMemoryBar({ layout }: { readonly layout: MemoryDomainBarLayout }) {
  const { domain, modelBytes, leftSegments, rightSegment, freeBytes } = layout
  const visibleLeft = leftSegments.filter(segment => segment.bytes > 0)
  const hasRight = rightSegment !== null && rightSegment.bytes > 0
  const flushRight = hasRight && visibleLeft.length > 0 && freeBytes === 0
  return <div className="settings-memory-domain" data-memory-domain={domain.id}>
    <div className="settings-memory-header">
      <span className="settings-memory-domain-label">{domain.label}</span>
      <span className="settings-memory-total">
        {modelBytes !== null
          ? <><strong data-memory-bytes={modelBytes}>{formatMemoryCapacity(modelBytes)}</strong><span>model · {formatMemoryCapacity(domain.totalBytes)} total</span></>
          : <span>{formatMemoryCapacity(domain.totalBytes)} total</span>}
      </span>
    </div>
    <div className="settings-memory-bar" aria-hidden="true">
      {visibleLeft.map(segment => <span
        key={segment.key}
        className={`settings-memory-segment model ${segment.key}`}
        style={{ width: `${segment.widthPercent}%` }}
      />)}
      {hasRight && <span
        className={`settings-memory-segment other${flushRight ? " flush" : ""}`}
        style={{ width: `${rightSegment.widthPercent}%` }}
      />}
    </div>
    <div className="settings-memory-legend">
      {domain.status === "inconsistent"
        ? <span className="settings-memory-notice">{domain.notice}</span>
        : <>
          <div className="settings-memory-parts">
            {leftSegments.map(segment => <span key={segment.key} className="settings-memory-item">
              <span aria-hidden="true" className={`settings-memory-key ${segment.key}`} />
              <span>{segment.label}</span>
              <strong data-memory-category={segment.label} data-bytes={segment.bytes}>{formatMemoryCapacity(segment.bytes)}</strong>
            </span>)}
            {freeBytes !== null && <span className="settings-memory-item">
              <span aria-hidden="true" className="settings-memory-key free" />
              <span>Free</span>
              <strong data-memory-category="Free" data-bytes={freeBytes}>{formatMemoryCapacity(freeBytes)}</strong>
            </span>}
            {rightSegment !== null && <span className="settings-memory-item">
              <span aria-hidden="true" className="settings-memory-key other" />
              <span>{rightSegment.label}</span>
              <strong data-memory-category={rightSegment.label} data-bytes={rightSegment.bytes}>{formatMemoryCapacity(rightSegment.bytes)}</strong>
            </span>}
          </div>
          {domain.notice !== null && <span className="settings-memory-notice">{domain.notice}</span>}
        </>}
    </div>
  </div>
}

function StreamingSummaryRow({ status }: { readonly status: ModelStreamingStatus }) {
  const summary = streamingResidencySummary(status)
  return <div className="settings-memory-streaming" data-streaming-summary="">
    <span className="settings-memory-streaming-share">
      <span>Experts in memory</span>
      <strong>{summary.primary}</strong>
      <span className="settings-memory-streaming-range">{summary.range}</span>
      {summary.secondary !== null && <span className="settings-memory-streaming-secondary">· {summary.secondary}</span>}
    </span>
    <span className="settings-memory-streaming-bytes">{summary.breakdown}</span>
  </div>
}

function ModelOnlyMemoryBar({ allocation }: { readonly allocation: Option.Option<ModelInstanceAllocation> }) {
  const domains = Option.match(allocation, { onNone: () => [], onSome: value => value.memoryDomains })
  const segments = [
    { label: "Weights", bytes: domains.reduce((sum, domain) => sum + domain.modelBytes + domain.auxiliaryBytes, 0), color: "weights" },
    { label: "Context", bytes: domains.reduce((sum, domain) => sum + domain.contextBytes, 0), color: "cache" },
    { label: "Overhead", bytes: domains.reduce((sum, domain) => sum + domain.computeBytes, 0), color: "runtime" },
  ]
  const total = segments.reduce((sum, segment) => sum + segment.bytes, 0)
  return <>
    <div className="settings-memory-bar" aria-hidden="true">
      {segments.filter(segment => segment.bytes > 0).map(segment => <span
        key={segment.label}
        className={`settings-memory-segment model ${segment.color}`}
        style={{ width: `${total > 0 ? segment.bytes / total * 100 : 0}%` }}
      />)}
    </div>
    <div className="settings-memory-legend">
      <span className="settings-memory-total">
        <span>Model memory</span>
        <strong data-memory-bytes={total}>{formatMemoryCapacity(total)}</strong>
      </span>
      <div className="settings-memory-parts">
        {segments.map(segment => <span key={segment.label} className="settings-memory-item">
          <span aria-hidden="true" className={`settings-memory-key ${segment.color}`} />
          <span>{segment.label}</span>
          <strong data-memory-category={segment.label} data-bytes={segment.bytes}>{formatMemoryCapacity(segment.bytes)}</strong>
        </span>)}
      </div>
    </div>
  </>
}

export function MemoryFigures({ allocation, hardware, reading = false, pendingLabel = "Available once loaded" }: {
  readonly allocation: Option.Option<ModelInstanceAllocation> | null
  readonly hardware: Option.Option<LocalInferenceHardware>
  readonly reading?: boolean
  readonly pendingLabel?: string
}) {
  if (allocation === null) {
    return <div className="settings-memory">
      <div className="settings-memory-bar" aria-hidden="true">
        {reading && <Skeleton className="h-full w-full" />}
      </div>
      <div className="settings-memory-legend">
        <span className="settings-memory-total">
          <span>Model memory</span>
          {reading ? <SkeletonLine className="h-4 w-16" /> : <span className="settings-memory-pending">{pendingLabel}</span>}
        </span>
      </div>
    </div>
  }

  const serving = Option.getOrNull(allocation)
  const streaming = serving === null ? null : Option.getOrNull(serving.streaming)
  const layouts = Option.match(hardware, {
    onNone: () => null,
    onSome: hw => participatingMemoryDomains(deriveHardwareMemoryView(hw, allocation)).map(memoryDomainBarLayout),
  })

  return <div className="settings-memory">
    {layouts !== null && layouts.length > 0
      ? <div className="settings-memory-domains">
        {layouts.map(layout => <DomainMemoryBar key={layout.domain.id} layout={layout} />)}
      </div>
      : <ModelOnlyMemoryBar allocation={allocation} />}
    {streaming !== null && <StreamingSummaryRow status={streaming} />}
  </div>
}
