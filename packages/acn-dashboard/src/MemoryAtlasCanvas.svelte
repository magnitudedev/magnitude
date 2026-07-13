<script lang="ts">
  import { onMount } from 'svelte'
  import type {
    AddressedAtlasGroup,
    AddressedAtlasNode,
    AddressedAtlasResident,
    AddressedAtlasSegment,
    AddressedPin,
    ProjectionIntrospection,
  } from './lib/types'

  interface Rect {
    readonly x: number
    readonly y: number
    readonly width: number
    readonly height: number
  }

  interface Tooltip {
    readonly x: number
    readonly y: number
    readonly title: string
    readonly lines: readonly string[]
  }

  type AtlasChild =
    | {
        readonly kind: 'state'
        readonly projectionName: string
        readonly bytes: number | null
        readonly weight: number
      }
    | {
        readonly kind: 'addressed'
        readonly node: AddressedAtlasNode
        readonly weight: number
      }

  interface AtlasProjection {
    readonly projection: ProjectionIntrospection | null
    readonly name: string
    readonly label: string
    readonly bytes: number | null
    readonly weight: number
    readonly children: readonly AtlasChild[]
  }

  type HitRegion =
    | {
        readonly kind: 'projection'
        readonly rect: Rect
        readonly projectionName: string
        readonly navDepth: number
      }
    | {
        readonly kind: 'state'
        readonly rect: Rect
        readonly projectionName: string
        readonly navDepth: number
      }
    | {
        readonly kind: 'addressed'
        readonly rect: Rect
        readonly node: AddressedAtlasNode
        readonly navDepth: number
      }
    | {
        readonly kind: 'control'
        readonly rect: Rect
        readonly action: 'root' | 'parent'
        readonly navDepth: number
      }

  type AtlasFocus =
    | { readonly kind: 'root' }
    | { readonly kind: 'projection'; readonly projectionName: string }
    | { readonly kind: 'addressed'; readonly path: readonly string[] }

  type HoverTarget =
    | { readonly kind: 'projection'; readonly projectionName: string }
    | { readonly kind: 'state'; readonly projectionName: string }
    | { readonly kind: 'addressed'; readonly path: readonly string[] }
    | { readonly kind: 'control'; readonly action: 'root' | 'parent' }

  let {
    projections = [],
    addressedAtlas = [],
    activeClientId = null,
    onSelectNode = () => {},
    onSelectProjection = () => {},
  } = $props<{
    readonly projections: readonly ProjectionIntrospection[]
    readonly addressedAtlas: readonly AddressedAtlasNode[]
    readonly activeClientId: string | null
    readonly onSelectNode?: (node: AddressedAtlasNode) => void
    readonly onSelectProjection?: (projectionName: string) => void
  }>()

  let wrapper: HTMLDivElement | undefined
  let canvas: HTMLCanvasElement | undefined
  let viewportWidth = $state(0)
  let viewportHeight = $state(0)
  let tooltip = $state<Tooltip | null>(null)
  let focus = $state<AtlasFocus>({ kind: 'root' })
  let hoverTarget = $state<HoverTarget | null>(null)
  let hitRegions: HitRegion[] = []

  const palette = {
    background: '#101114',
    panel: '#17191f',
    panel2: '#1e2129',
    border: '#30343f',
    muted: '#8e96a8',
    text: '#e7e9ee',
    green: '#5bd49b',
    residentFill: 'rgba(91, 212, 155, 0.58)',
    residentStroke: 'rgba(105, 230, 168, 0.9)',
    blue: '#5ca7ff',
    stateFill: 'rgba(92, 167, 255, 0.32)',
    stateStroke: 'rgba(116, 181, 255, 0.72)',
    yellow: '#f3c969',
    purple: '#c38cff',
    offloadedFill: '#22252c',
    offloadedStroke: '#464b57',
  }

  const formatBytes = (bytes: number): string => {
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${bytes} B`
  }

  const shortId = (value: string): string =>
    value.length > 14 ? value.slice(0, 14) : value

  const projectionLabel = (projection: ProjectionIntrospection): string =>
    projection.name.replace(/Projection$/, '')

  const projectionSummary = (projection: ProjectionIntrospection): string => {
    if (projection.summary?.label) return projection.summary.label
    const state = projection.state
    if (state == null) return 'empty'
    if (typeof state !== 'object') return typeof state
    if (Array.isArray(state)) return `${state.length} items`
    return `${Object.keys(state as Record<string, unknown>).length} keys`
  }

  const stateBytes = (projection: ProjectionIntrospection): number | null =>
    projection.summary?.estimatedBytes ?? null

  const nodeWeight = (node: AddressedAtlasNode): number =>
    Math.max(1, node.bytes || node.residentEntryCount + node.offloadedEntryCount || 1)

  const isLayoutOnlyGroup = (node: AddressedAtlasNode): node is AddressedAtlasGroup =>
    node.kind === 'group' && (
      node.role === 'projection' ||
      node.role === 'collection' ||
      node.role === 'branch'
    )

  const visibleAddressedNode = (node: AddressedAtlasNode): AddressedAtlasNode => {
    let current = node
    while (isLayoutOnlyGroup(current) && current.children.length === 1) {
      current = current.children[0]
    }
    return current
  }

  const pinClientId = (pin: AddressedPin): string | null => {
    if (pin.kind !== 'display-view') return null
    return pin.viewId ?? pin.owner.replace(/^display-view:/, '')
  }

  const displayPins = (node: AddressedAtlasSegment | AddressedAtlasResident): readonly AddressedPin[] =>
    node.pins.filter((pin) => pin.kind === 'display-view')

  const producerPins = (node: AddressedAtlasSegment | AddressedAtlasResident): readonly AddressedPin[] =>
    node.pins.filter((pin) => pin.kind === 'display-producer')

  const nodePinnedByClient = (
    node: AddressedAtlasNode,
    clientId: string | null,
  ): boolean => {
    if (clientId === null) return false
    if (node.kind === 'group') return node.children.some((child) => nodePinnedByClient(child, clientId))
    return displayPins(node).some((pin) => pinClientId(pin) === clientId)
  }

  const segmentRange = (segment: AddressedAtlasSegment): string => {
    if (segment.itemCount <= 0) return `${segment.startOffset}`
    const end = segment.startOffset + segment.itemCount - 1
    return segment.startOffset === end ? `${segment.startOffset}` : `${segment.startOffset}-${end}`
  }

  const atlasRootsByProjection = (): Map<string, AddressedAtlasNode[]> => {
    const map = new Map<string, AddressedAtlasNode[]>()
    for (const node of addressedAtlas) {
      map.set(node.projection, [...(map.get(node.projection) ?? []), node])
    }
    return map
  }

  const atlasProjections = (): AtlasProjection[] => {
    const rootsByProjection = atlasRootsByProjection()
    const projectionNames = new Set<string>(projections.map((projection: ProjectionIntrospection) => projection.name))
    for (const name of rootsByProjection.keys()) projectionNames.add(name)

    const rows = [...projectionNames].map((name) => {
      const projection = projections.find((candidate: ProjectionIntrospection) => candidate.name === name) ?? null
      const roots = rootsByProjection.get(name) ?? []
      const stateSize = projection ? stateBytes(projection) : null
      const children: AtlasChild[] = [
        ...(projection
          ? [{
              kind: 'state' as const,
              projectionName: projection.name,
              bytes: stateSize,
              weight: Math.max(1, stateSize ?? 1),
            }]
          : []),
        ...roots.map((node): AtlasChild => ({
          kind: 'addressed',
          node,
          weight: nodeWeight(node),
        })),
      ]
      const bytes = (stateSize ?? 0) + roots.reduce((sum, node) => sum + node.bytes, 0)
      const weight = children.reduce((sum, child) => sum + child.weight, 0)

      return {
        projection,
        name,
        label: projection ? projectionLabel(projection) : name.replace(/Projection$/, ''),
        bytes: bytes > 0 ? bytes : null,
        weight: Math.max(1, weight),
        children,
      }
    })

    return rows.sort((left, right) => {
      if (left.label === 'DisplayTimeline') return -1
      if (right.label === 'DisplayTimeline') return 1
      return right.weight - left.weight || left.label.localeCompare(right.label)
    })
  }

  const samePath = (left: readonly string[], right: readonly string[]): boolean =>
    left.length === right.length && left.every((part, index) => part === right[index])

  const findAddressedNode = (
    nodes: readonly AddressedAtlasNode[],
    path: readonly string[],
  ): AddressedAtlasNode | null => {
    for (const node of nodes) {
      if (samePath(node.path, path)) return node
      if (node.kind === 'group') {
        const child = findAddressedNode(node.children, path)
        if (child) return child
      }
    }
    return null
  }

  const visualAddressedParent = (path: readonly string[]): AtlasFocus => {
    let parent = path.slice(0, -1)
    while (parent.length > 0) {
      const node = findAddressedNode(addressedAtlas, parent)
      if (node && !isLayoutOnlyGroup(node)) return { kind: 'addressed', path: parent }
      parent = parent.slice(0, -1)
    }
    return path[0] ? { kind: 'projection', projectionName: path[0] } : { kind: 'root' }
  }

  const focusParent = (): AtlasFocus => {
    if (focus.kind === 'projection') return { kind: 'root' }
    if (focus.kind !== 'addressed') return { kind: 'root' }
    return visualAddressedParent(focus.path)
  }

  const effectiveFocus = (atlas: readonly AtlasProjection[]): AtlasFocus => {
    const currentFocus = focus
    if (currentFocus.kind === 'projection') {
      return atlas.some((candidate) => candidate.name === currentFocus.projectionName)
        ? currentFocus
        : { kind: 'root' }
    }
    if (currentFocus.kind === 'addressed') {
      return findAddressedNode(addressedAtlas, currentFocus.path)
        ? currentFocus
        : { kind: 'root' }
    }
    return currentFocus
  }

  const focusLabel = (currentFocus: AtlasFocus): string => {
    if (currentFocus.kind === 'root') return 'all memory'
    if (currentFocus.kind === 'projection') return currentFocus.projectionName.replace(/Projection$/, '')
    return currentFocus.path.join(' / ')
  }

  const splitLayout = <A extends { readonly weight: number }>(
    items: readonly A[],
    rect: Rect,
  ): Array<{ readonly item: A; readonly rect: Rect }> => {
    if (items.length === 0 || rect.width <= 0 || rect.height <= 0) return []
    if (items.length === 1) return [{ item: items[0], rect }]

    const sorted = [...items].sort((left, right) => right.weight - left.weight)
    const total = sorted.reduce((sum, item) => sum + item.weight, 0)
    const half = total / 2
    let leftWeight = 0
    let splitIndex = 0
    while (splitIndex < sorted.length - 1 && leftWeight + sorted[splitIndex].weight <= half) {
      leftWeight += sorted[splitIndex].weight
      splitIndex += 1
    }
    if (splitIndex === 0) {
      leftWeight = sorted[0].weight
      splitIndex = 1
    }

    const first = sorted.slice(0, splitIndex)
    const second = sorted.slice(splitIndex)
    const ratio = total <= 0 ? 0.5 : leftWeight / total

    if (rect.width >= rect.height) {
      const leftWidth = rect.width * ratio
      return [
        ...splitLayout(first, { ...rect, width: leftWidth }),
        ...splitLayout(second, {
          x: rect.x + leftWidth,
          y: rect.y,
          width: rect.width - leftWidth,
          height: rect.height,
        }),
      ]
    }

    const topHeight = rect.height * ratio
    return [
      ...splitLayout(first, { ...rect, height: topHeight }),
      ...splitLayout(second, {
        x: rect.x,
        y: rect.y + topHeight,
        width: rect.width,
        height: rect.height - topHeight,
      }),
    ]
  }

  const inset = (rect: Rect, amount: number): Rect => ({
    x: rect.x + amount,
    y: rect.y + amount,
    width: Math.max(0, rect.width - amount * 2),
    height: Math.max(0, rect.height - amount * 2),
  })

  const roundedRect = (
    context: CanvasRenderingContext2D,
    rect: Rect,
    radius: number,
  ) => {
    context.beginPath()
    context.roundRect(rect.x, rect.y, rect.width, rect.height, radius)
  }

  const drawText = (
    context: CanvasRenderingContext2D,
    text: string,
    x: number,
    y: number,
    color = palette.text,
    font = '12px ui-monospace, SFMono-Regular, Menlo, monospace',
  ) => {
    context.fillStyle = color
    context.font = font
    context.fillText(text, x, y)
  }

  const sameHoverTarget = (left: HoverTarget | null, right: HoverTarget): boolean => {
    if (!left || left.kind !== right.kind) return false
    if (left.kind === 'projection' && right.kind === 'projection') {
      return left.projectionName === right.projectionName
    }
    if (left.kind === 'state' && right.kind === 'state') {
      return left.projectionName === right.projectionName
    }
    if (left.kind === 'addressed' && right.kind === 'addressed') {
      return samePath(left.path, right.path)
    }
    if (left.kind === 'control' && right.kind === 'control') {
      return left.action === right.action
    }
    return false
  }

  const regionTarget = (region: HitRegion): HoverTarget => {
    if (region.kind === 'projection') {
      return { kind: 'projection', projectionName: region.projectionName }
    }
    if (region.kind === 'state') {
      return { kind: 'state', projectionName: region.projectionName }
    }
    if (region.kind === 'addressed') {
      return { kind: 'addressed', path: region.node.path }
    }
    return { kind: 'control', action: region.action }
  }

  const drawHoverOutline = (
    context: CanvasRenderingContext2D,
    rect: Rect,
    radius: number,
  ) => {
    context.save()
    roundedRect(context, rect, radius)
    context.lineWidth = 2.5
    context.strokeStyle = 'rgba(231, 233, 238, 0.92)'
    context.shadowColor = 'rgba(92, 167, 255, 0.7)'
    context.shadowBlur = 8
    context.stroke()
    context.restore()
  }

  const drawLegend = (context: CanvasRenderingContext2D, rect: Rect) => {
    const items = [
      ['state', 'state'],
      ['resident', 'resident'],
      ['offloaded', 'offloaded'],
      ['consumer pin', 'consumer'],
      ['producer pin', 'producer'],
      ['dirty', 'dirty'],
    ] as const

    let x = rect.x
    for (const [label, kind] of items) {
      context.lineWidth = 1.5
      context.setLineDash([])
      context.fillStyle = kind === 'state'
        ? palette.stateFill
        : kind === 'resident'
          ? palette.residentFill
          : kind === 'offloaded'
            ? palette.offloadedFill
            : palette.panel2
      context.strokeStyle = kind === 'state'
        ? palette.stateStroke
        : kind === 'resident'
          ? palette.residentStroke
          : kind === 'dirty'
            ? palette.yellow
            : palette.offloadedStroke
      context.fillRect(x, rect.y, 13, 10)
      context.strokeRect(x, rect.y, 13, 10)
      if (kind === 'consumer') {
        context.fillStyle = palette.blue
        context.fillRect(x, rect.y + 7, 13, 3)
      }
      if (kind === 'producer') {
        context.fillStyle = palette.purple
        context.fillRect(x + 10, rect.y, 3, 10)
      }
      drawText(context, label, x + 15, rect.y + 9, palette.muted, '11px Inter, sans-serif')
      x += label.length * 6 + 34
    }
  }

  const drawFocusBar = (
    context: CanvasRenderingContext2D,
    rect: Rect,
    currentFocus: AtlasFocus,
  ) => {
    if (currentFocus.kind === 'root') return

    const allRect = { x: rect.x, y: rect.y, width: 42, height: 22 }
    const backRect = { x: rect.x + 48, y: rect.y, width: 50, height: 22 }
    hitRegions.push({ kind: 'control', rect: allRect, action: 'root', navDepth: 0 })
    hitRegions.push({ kind: 'control', rect: backRect, action: 'parent', navDepth: 0 })

    for (const [label, buttonRect] of [['all', allRect], ['back', backRect]] as const) {
      roundedRect(context, buttonRect, 5)
      context.fillStyle = palette.panel2
      context.fill()
      context.strokeStyle = palette.border
      context.stroke()
      if (sameHoverTarget(hoverTarget, { kind: 'control', action: label === 'all' ? 'root' : 'parent' })) {
        drawHoverOutline(context, buttonRect, 5)
      }
      drawText(context, label, buttonRect.x + 9, buttonRect.y + 15, palette.text, '11px Inter, sans-serif')
    }

    drawText(
      context,
      focusLabel(currentFocus),
      rect.x + 112,
      rect.y + 15,
      palette.text,
      '600 12px Inter, sans-serif',
    )
  }

  const drawState = (
    context: CanvasRenderingContext2D,
    child: Extract<AtlasChild, { readonly kind: 'state' }>,
    rect: Rect,
    label = 'state',
  ) => {
    roundedRect(context, rect, 5)
    context.fillStyle = palette.stateFill
    context.fill()
    context.strokeStyle = palette.stateStroke
    context.stroke()
    if (sameHoverTarget(hoverTarget, { kind: 'state', projectionName: child.projectionName })) {
      drawHoverOutline(context, rect, 5)
    }
    if (rect.width > 74 && rect.height > 34) {
      drawText(context, label, rect.x + 8, rect.y + 17, palette.text)
      if (child.bytes !== null) {
        drawText(context, formatBytes(child.bytes), rect.x + 8, rect.y + 32, palette.muted, '11px ui-monospace, SFMono-Regular, Menlo, monospace')
      }
    }
  }

  const drawSegment = (
    context: CanvasRenderingContext2D,
    segment: AddressedAtlasSegment,
    rect: Rect,
  ) => {
    const resident = segment.residency === 'resident'
    const active = nodePinnedByClient(segment, activeClientId)
    const dimmed = activeClientId !== null && !active

    context.save()
    context.globalAlpha = dimmed ? 0.22 : 1
    roundedRect(context, rect, 5)
    context.fillStyle = resident ? palette.residentFill : palette.offloadedFill
    context.fill()
    context.lineWidth = segment.dirty ? 3 : 1
    context.strokeStyle = segment.dirty
      ? palette.yellow
      : resident
        ? palette.residentStroke
        : palette.offloadedStroke
    context.stroke()

    if (displayPins(segment).length > 0) {
      context.fillStyle = active ? palette.blue : 'rgba(92, 167, 255, 0.72)'
      context.fillRect(rect.x, rect.y + rect.height - 5, rect.width, 5)
    }
    if (producerPins(segment).length > 0) {
      context.fillStyle = palette.purple
      context.fillRect(rect.x + rect.width - 7, rect.y, 7, rect.height)
    }
    if (sameHoverTarget(hoverTarget, { kind: 'addressed', path: segment.path })) {
      drawHoverOutline(context, rect, 5)
    }

    if (rect.width > 72 && rect.height > 36) {
      drawText(context, segment.logicalSegmentId, rect.x + 8, rect.y + 17, palette.text)
      drawText(
        context,
        `${segmentRange(segment)} · ${segment.estimatedBytes === null ? 'unknown' : formatBytes(segment.estimatedBytes)}`,
        rect.x + 8,
        rect.y + 32,
        palette.muted,
        '11px ui-monospace, SFMono-Regular, Menlo, monospace',
      )
    }
    context.restore()
  }

  const drawResident = (
    context: CanvasRenderingContext2D,
    resident: AddressedAtlasResident,
    rect: Rect,
  ) => {
    const active = nodePinnedByClient(resident, activeClientId)
    const dimmed = activeClientId !== null && !active

    context.save()
    context.globalAlpha = dimmed ? 0.22 : 1
    roundedRect(context, rect, 5)
    context.fillStyle = palette.residentFill
    context.fill()
    context.strokeStyle = resident.dirty ? palette.yellow : palette.residentStroke
    context.stroke()
    if (displayPins(resident).length > 0) {
      context.fillStyle = active ? palette.blue : 'rgba(92, 167, 255, 0.72)'
      context.fillRect(rect.x, rect.y + rect.height - 5, rect.width, 5)
    }
    if (sameHoverTarget(hoverTarget, { kind: 'addressed', path: resident.path })) {
      drawHoverOutline(context, rect, 5)
    }
    if (rect.width > 84 && rect.height > 34) {
      drawText(context, 'resident', rect.x + 8, rect.y + 17, palette.text)
      drawText(context, formatBytes(resident.estimatedResidentBytes), rect.x + 8, rect.y + 32, palette.muted, '11px ui-monospace, SFMono-Regular, Menlo, monospace')
    }
    context.restore()
  }

  const groupHeader = (group: AddressedAtlasGroup): string => {
    if (group.role === 'fork') return group.label === 'root' ? 'root fork' : shortId(group.label)
    if (group.role === 'collection') return group.label
    return group.label
  }

  const drawAddressedNode = (
    context: CanvasRenderingContext2D,
    node: AddressedAtlasNode,
    rect: Rect,
    depth: number,
    navDepth: number,
  ) => {
    const visible = visibleAddressedNode(node)
    if (visible !== node) {
      drawAddressedNode(context, visible, rect, depth, navDepth)
      return
    }

    if (isLayoutOnlyGroup(node)) {
      const children = node.children.map(visibleAddressedNode).map((child) => ({
        node: child,
        weight: nodeWeight(child),
      }))
      for (const { item, rect: childRect } of splitLayout(children, rect)) {
        drawAddressedNode(context, item.node, childRect, depth, navDepth)
      }
      return
    }

    const outer = inset(rect, Math.max(1, 3 - Math.min(depth, 2)))
    if (outer.width < 4 || outer.height < 4) return

    hitRegions.push({ kind: 'addressed', rect: outer, node, navDepth })

    if (node.kind === 'segment') {
      drawSegment(context, node, outer)
      return
    }

    if (node.kind === 'resident') {
      drawResident(context, node, outer)
      return
    }

    const group = node as AddressedAtlasGroup

    roundedRect(context, outer, depth === 0 ? 7 : 5)
    context.fillStyle = depth === 0 ? palette.panel2 : 'rgba(23, 25, 31, 0.56)'
    context.fill()
    context.strokeStyle = depth === 0 ? 'rgba(91, 212, 155, 0.34)' : palette.border
    context.stroke()

    const body = inset(outer, depth === 0 ? 5 : 3)

    const children = group.children.map(visibleAddressedNode).map((child: AddressedAtlasNode) => ({
      node: child,
      weight: nodeWeight(child),
    }))

    for (const { item, rect: childRect } of splitLayout(children, body)) {
      drawAddressedNode(context, item.node, childRect, depth + 1, navDepth + 1)
    }

    if (outer.width > 96 && outer.height > 54) {
      context.fillStyle = 'rgba(23, 25, 31, 0.72)'
      context.fillRect(outer.x + 6, outer.y + 6, Math.min(outer.width - 12, 170), 23)
      drawText(context, groupHeader(group), outer.x + 10, outer.y + 18, palette.text, '700 12px Inter, sans-serif')
      drawText(
        context,
        group.bytes > 0 ? formatBytes(group.bytes) : `${group.residentEntryCount + group.offloadedEntryCount} entries`,
        outer.x + 10,
        outer.y + 28,
        palette.muted,
        '10px ui-monospace, SFMono-Regular, Menlo, monospace',
      )
    }
    if (sameHoverTarget(hoverTarget, { kind: 'addressed', path: group.path })) {
      drawHoverOutline(context, outer, depth === 0 ? 7 : 5)
    }
  }

  const drawProjection = (
    context: CanvasRenderingContext2D,
    atlas: AtlasProjection,
    rect: Rect,
    navDepth: number,
  ) => {
    const outer = inset(rect, 4)
    if (outer.width < 12 || outer.height < 12) return

    hitRegions.push({ kind: 'projection', rect: outer, projectionName: atlas.name, navDepth })

    if (atlas.children.length === 1 && atlas.children[0].kind === 'state') {
      drawState(context, atlas.children[0], outer, atlas.label)
      if (sameHoverTarget(hoverTarget, { kind: 'projection', projectionName: atlas.name })) {
        drawHoverOutline(context, outer, 5)
      }
      return
    }

    roundedRect(context, outer, 8)
    context.fillStyle = palette.panel
    context.fill()
    context.strokeStyle = palette.border
    context.stroke()

    const body = inset(outer, 6)

    for (const { item, rect: childRect } of splitLayout(atlas.children, body)) {
      const childOuter = inset(childRect, 2)
      if (childOuter.width < 4 || childOuter.height < 4) continue
      drawAtlasChild(context, item, childOuter, 0, navDepth + 1)
    }

    if (outer.width > 108 && outer.height > 62) {
      context.fillStyle = 'rgba(23, 25, 31, 0.76)'
      context.fillRect(outer.x + 8, outer.y + 7, Math.min(outer.width - 16, 190), 27)
      drawText(context, atlas.label, outer.x + 12, outer.y + 20, palette.text, '700 13px Inter, sans-serif')
      drawText(
        context,
        atlas.bytes === null ? 'size unknown' : formatBytes(atlas.bytes),
        outer.x + 12,
        outer.y + 32,
        palette.muted,
        '11px ui-monospace, SFMono-Regular, Menlo, monospace',
      )
    }
    if (sameHoverTarget(hoverTarget, { kind: 'projection', projectionName: atlas.name })) {
      drawHoverOutline(context, outer, 8)
    }
  }

  const drawAtlasChild = (
    context: CanvasRenderingContext2D,
    child: AtlasChild,
    rect: Rect,
    depth: number,
    navDepth: number,
  ) => {
    if (rect.width < 4 || rect.height < 4) return
    if (child.kind === 'state') {
      hitRegions.push({ kind: 'state', rect, projectionName: child.projectionName, navDepth })
      drawState(context, child, rect)
      return
    }
    drawAddressedNode(context, child.node, rect, depth, navDepth)
  }

  const drawFocusedProjection = (
    context: CanvasRenderingContext2D,
    atlas: AtlasProjection,
    rect: Rect,
  ) => {
    if (atlas.children.length === 0) return
    for (const { item, rect: childRect } of splitLayout(atlas.children, rect)) {
      drawAtlasChild(context, item, inset(childRect, 3), 0, 1)
    }
  }

  const drawFocusedAddressed = (
    context: CanvasRenderingContext2D,
    node: AddressedAtlasNode,
    rect: Rect,
  ) => {
    drawAddressedNode(context, node, inset(rect, 3), 0, 0)
  }

  const drawAtlas = () => {
    if (!canvas || viewportWidth <= 0 || viewportHeight <= 0) return

    const context = canvas.getContext('2d')
    if (!context) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.max(1, Math.floor(viewportWidth * dpr))
    canvas.height = Math.max(1, Math.floor(viewportHeight * dpr))
    canvas.style.width = `${viewportWidth}px`
    canvas.style.height = `${viewportHeight}px`
    context.setTransform(dpr, 0, 0, dpr, 0, 0)

    hitRegions = []
    const atlas = atlasProjections()
    const currentFocus = effectiveFocus(atlas)
    context.fillStyle = palette.background
    context.fillRect(0, 0, viewportWidth, viewportHeight)

    drawLegend(context, { x: 18, y: 18, width: viewportWidth - 36, height: 12 })
    drawFocusBar(context, { x: 18, y: 39, width: viewportWidth - 36, height: 22 }, currentFocus)

    const rect = {
      x: 14,
      y: currentFocus.kind === 'root' ? 46 : 72,
      width: Math.max(1, viewportWidth - 28),
      height: Math.max(1, viewportHeight - (currentFocus.kind === 'root' ? 60 : 86)),
    }

    if (currentFocus.kind === 'projection') {
      const selected = atlas.find((candidate) => candidate.name === currentFocus.projectionName)
      if (selected) {
        drawFocusedProjection(context, selected, rect)
        return
      }
    }

    if (currentFocus.kind === 'addressed') {
      const selected = findAddressedNode(addressedAtlas, currentFocus.path)
      if (selected) {
        drawFocusedAddressed(context, selected, rect)
        return
      }
    }

    for (const layout of splitLayout(atlas, rect)) {
      drawProjection(context, layout.item, layout.rect, 1)
    }
  }

  const contains = (rect: Rect, x: number, y: number): boolean =>
    x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height

  const regionAt = (x: number, y: number): HitRegion | null => {
    const contained = hitRegions.filter((region) => contains(region.rect, x, y))
    if (contained.length === 0) return null

    const control = [...contained].reverse().find((region) => region.kind === 'control')
    if (control) return control

    const positiveDepth = contained.filter((region) => region.navDepth > 0)
    const candidates = positiveDepth.length > 0 ? positiveDepth : contained
    const depth = Math.min(...candidates.map((region) => region.navDepth))
    return [...candidates].reverse().find((region) => region.navDepth === depth) ?? null
  }

  const addressedTooltip = (
    node: AddressedAtlasNode,
    x: number,
    y: number,
  ): Tooltip => {
    if (node.kind === 'group') {
      return {
        x,
        y,
        title: node.path.join(' / '),
        lines: [
          `${node.bytes > 0 ? formatBytes(node.bytes) : 'unknown size'}`,
          `${node.residentEntryCount} loaded · ${node.offloadedEntryCount} offloaded`,
          `${node.pinnedEntryCount} pinned · ${node.dirtyEntryCount} dirty`,
        ],
      }
    }

    if (node.kind === 'resident') {
      const pins = node.pins.length === 0
        ? 'no pins'
        : node.pins.map((pin) => pinClientId(pin) ?? pin.owner).join(', ')
      return {
        x,
        y,
        title: node.path.join(' / '),
        lines: [
          `resident · ${formatBytes(node.estimatedResidentBytes)}`,
          shortId(node.address),
          pins,
        ],
      }
    }

    const pins = node.pins.length === 0
      ? 'no pins'
      : node.pins.map((pin) => pinClientId(pin) ?? pin.owner).join(', ')
    return {
      x,
      y,
      title: node.path.join(' / '),
      lines: [
        `${node.residency}${node.dirty ? ' dirty' : ''} · ${node.estimatedBytes === null ? 'size unknown' : formatBytes(node.estimatedBytes)}`,
        `${segmentRange(node)} · ${node.itemCount} items`,
        shortId(node.address),
        pins,
      ],
    }
  }

  const projectionTooltip = (projectionName: string, x: number, y: number): Tooltip => {
    const projection = projections.find((candidate: ProjectionIntrospection) => candidate.name === projectionName)
    return {
      x,
      y,
      title: projectionName.replace(/Projection$/, ''),
      lines: projection
        ? [
            projection.kind,
            projectionSummary(projection),
            projection.summary?.estimatedBytes ? formatBytes(projection.summary.estimatedBytes) : 'state size unknown',
          ]
        : ['addressed-only projection'],
    }
  }

  const handleMove = (event: MouseEvent) => {
    const region = regionAt(event.offsetX, event.offsetY)
    if (!region) {
      tooltip = null
      hoverTarget = null
      if (canvas) canvas.style.cursor = 'default'
      return
    }

    if (canvas) canvas.style.cursor = 'pointer'
    hoverTarget = regionTarget(region)
    if (region.kind === 'control') {
      tooltip = null
      return
    }
    tooltip = region.kind === 'addressed'
      ? addressedTooltip(region.node, event.offsetX + 14, event.offsetY + 14)
      : projectionTooltip(region.projectionName, event.offsetX + 14, event.offsetY + 14)
  }

  const handleClick = (event: MouseEvent) => {
    const region = regionAt(event.offsetX, event.offsetY)
    if (!region) return

    if (region.kind === 'control') {
      focus = region.action === 'root' ? { kind: 'root' } : focusParent()
      return
    }

    if (region.kind === 'addressed') {
      focus = { kind: 'addressed', path: region.node.path }
      onSelectNode(region.node)
      return
    }

    if (region.kind === 'projection') {
      focus = { kind: 'projection', projectionName: region.projectionName }
    }
    onSelectProjection(region.projectionName)
  }

  onMount(() => {
    if (!wrapper) return
    const observer = new ResizeObserver(([entry]) => {
      viewportWidth = Math.max(1, Math.floor(entry.contentRect.width))
      viewportHeight = Math.max(1, Math.floor(entry.contentRect.height))
    })
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || focus.kind === 'root') return
      event.preventDefault()
      tooltip = null
      hoverTarget = null
      focus = focusParent()
    }
    observer.observe(wrapper)
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      observer.disconnect()
      window.removeEventListener('keydown', handleKeyDown)
    }
  })

  $effect(() => {
    projections
    addressedAtlas
    activeClientId
    focus
    hoverTarget
    viewportWidth
    viewportHeight
    drawAtlas()
  })
</script>

<div class="memory-canvas-wrap" bind:this={wrapper}>
  <canvas
    class="memory-canvas"
    bind:this={canvas}
    onmousemove={handleMove}
    onmouseleave={() => {
      tooltip = null
      hoverTarget = null
    }}
    onclick={handleClick}
  ></canvas>
  {#if tooltip}
    <div class="atlas-tooltip" style={`left: ${tooltip.x}px; top: ${tooltip.y}px`}>
      <strong>{tooltip.title}</strong>
      {#each tooltip.lines as line}
        <span>{line}</span>
      {/each}
    </div>
  {/if}
</div>

<style>
  .memory-canvas-wrap {
    position: relative;
    min-height: 0;
    height: 100%;
    overflow: hidden;
    background: #101114;
  }

  .memory-canvas {
    display: block;
  }

  .atlas-tooltip {
    position: absolute;
    z-index: 10;
    display: grid;
    gap: 4px;
    max-width: 360px;
    padding: 9px 10px;
    border: 1px solid #30343f;
    border-radius: 6px;
    background: rgba(23, 25, 31, 0.96);
    color: #e7e9ee;
    pointer-events: none;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  }

  .atlas-tooltip strong {
    font-size: 12px;
    overflow-wrap: anywhere;
  }

  .atlas-tooltip span {
    color: #8e96a8;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
    overflow-wrap: anywhere;
  }
</style>
