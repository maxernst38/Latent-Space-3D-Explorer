/**
 * The viewer component: everything around the canvas — class picker, overlay
 * toggles, outlier mode, tooltip, eigen-image strip. The drawing itself is
 * scene.js; this file decides what to draw and renders the chrome.
 *
 * Laid out to fill its container rather than to sit in a fixed-height card:
 * the canvas takes all the room left over, the controls live in a sidebar with
 * enough height for a real class list, and the eigen strip sits under the
 * canvas because it is a reading of the data rather than a control.
 *
 * Styling uses the semantic tokens declared in web/src/index.css. Note the
 * split: chrome off the canvas uses the light tokens (text-ink, bg-surface),
 * while anything drawn *over* the canvas uses the dark `canvas-*` ones.
 *
 * Every Tailwind class here is written out in full. Tailwind v4 never sees
 * classes assembled at runtime, so per-class colours come from inline styles.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { loadBundle } from './loadBundle.js'
import { highlightColor } from './palette.js'
import { createScene } from './scene.js'

const MAX_HIGHLIGHTS = 8

const CANVAS_BUTTON =
  'inline-flex h-8 items-center rounded-pill border border-canvas-hairline ' +
  'bg-canvas-panel px-3 text-xs font-medium text-canvas-ink backdrop-blur-sm ' +
  'hover:border-canvas-muted'

export default function Viewer({ bundleUrl }) {
  const canvasRef = useRef(null)
  const sceneRef = useRef(null)

  const [bundle, setBundle] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState([]) // class ids, in click order
  const [search, setSearch] = useState('')
  const [hovered, setHovered] = useState(-1)
  const [labels, setLabels] = useState([])
  const [tooltip, setTooltip] = useState(null)
  const [options, setOptions] = useState({
    showCentroids: true,
    showHulls: false,
    showAxes: false,
    outlierMode: false,
    spinning: true,
  })

  /* ── Data ───────────────────────────────────────────────────────────── */

  useEffect(() => {
    const controller = new AbortController()
    setBundle(null)
    setError(null)
    setSelected([])

    loadBundle(bundleUrl, { signal: controller.signal })
      .then(setBundle)
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message)
      })

    return () => controller.abort()
  }, [bundleUrl])

  /* ── Scene lifecycle ────────────────────────────────────────────────── */

  useEffect(() => {
    if (!canvasRef.current) return
    const scene = createScene(canvasRef.current)
    scene.onHover(setHovered)
    scene.onLabels(setLabels)
    sceneRef.current = scene

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setOptions((o) => ({ ...o, spinning: false }))
    }

    return () => {
      scene.dispose()
      sceneRef.current = null
    }
  }, [])

  useEffect(() => {
    if (bundle) sceneRef.current?.setBundle(bundle)
  }, [bundle])

  const highlights = useMemo(
    () => new Map(selected.map((id, i) => [id, highlightColor(i)])),
    [selected],
  )

  useEffect(() => {
    sceneRef.current?.setHighlights(highlights)
  }, [highlights])

  useEffect(() => {
    sceneRef.current?.setOptions(options)
  }, [options])

  /* ── Interaction ────────────────────────────────────────────────────── */

  // The canvas size travels with the pointer position so the hover card can
  // flip against the real edges; it no longer has a fixed height to assume.
  const onPointerMove = useCallback((event) => {
    sceneRef.current?.setPointer(event)
    const rect = event.currentTarget.getBoundingClientRect()
    setTooltip({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      w: rect.width,
      h: rect.height,
    })
  }, [])

  const onPointerLeave = useCallback(() => {
    sceneRef.current?.clearPointer()
    setTooltip(null)
  }, [])

  const toggleClass = useCallback((id) => {
    setSelected((current) =>
      current.includes(id)
        ? current.filter((c) => c !== id)
        : current.length >= MAX_HIGHLIGHTS
          ? current
          : [...current, id],
    )
  }, [])

  const toggleOption = useCallback((key) => {
    setOptions((current) => ({ ...current, [key]: !current[key] }))
  }, [])

  /* ── Render ─────────────────────────────────────────────────────────── */

  if (error) {
    return (
      <div className="grid flex-1 place-items-center px-5">
        <div className="max-w-md text-center">
          <p className="font-medium text-ink">Could not load that dataset.</p>
          <p className="mt-2 text-sm text-body">{error}</p>
        </div>
      </div>
    )
  }

  const classes = bundle?.manifest.classes ?? []
  const visibleClasses = search
    ? classes.filter((c) => c.name.toLowerCase().includes(search.toLowerCase()))
    : classes
  const soloClass = selected.length === 1 ? classes[selected[0]] : null

  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="relative min-h-0 flex-1 bg-canvas">
          <canvas
            ref={canvasRef}
            onPointerMove={onPointerMove}
            onPointerLeave={onPointerLeave}
            className="absolute inset-0 block h-full w-full cursor-grab touch-none active:cursor-grabbing"
            role="img"
            aria-label={
              bundle
                ? `${bundle.manifest.count} images from ${bundle.manifest.name}, positioned in 3D by visual similarity. Drag to rotate.`
                : 'Loading the embedding view'
            }
          />

          {!bundle && (
            <div className="absolute inset-0 grid place-items-center" role="status">
              <div className="flex flex-col items-center gap-3">
                <span
                  className="h-6 w-6 animate-spin rounded-pill border-2 border-canvas-hairline border-t-canvas-ink"
                  aria-hidden="true"
                />
                <span className="text-sm text-canvas-muted">Loading embeddings…</span>
              </div>
            </div>
          )}

          {bundle && (
            <Stats manifest={bundle.manifest} outlierMode={options.outlierMode} />
          )}

          {labels.map((label) => (
            <span
              key={label.id}
              className="pointer-events-none absolute -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-input px-1.5 py-0.5 text-xs font-medium"
              style={{
                left: label.x,
                top: label.y - 10,
                color: label.color,
                background: 'rgb(24 28 35 / 0.78)',
              }}
            >
              {label.name}
            </span>
          ))}

          {bundle && tooltip && hovered >= 0 && (
            <HoverCard bundle={bundle} index={hovered} at={tooltip} />
          )}

          {bundle && (
            <div className="absolute bottom-3 right-3 flex gap-2">
              {soloClass?.centroid && (
                <button
                  type="button"
                  onClick={() => sceneRef.current?.focusClass(soloClass.id)}
                  className={CANVAS_BUTTON}
                >
                  Fly to class
                </button>
              )}
              <button
                type="button"
                onClick={() => sceneRef.current?.resetCamera()}
                className={CANVAS_BUTTON}
              >
                Reset view
              </button>
            </div>
          )}
        </div>

        {bundle && soloClass?.eigenImages && !options.outlierMode && (
          <EigenStrip
            bundle={bundle}
            entry={soloClass}
            color={highlights.get(soloClass.id)}
          />
        )}
      </div>

      <Sidebar
        loading={!bundle}
        classes={visibleClasses}
        allClasses={classes}
        selected={selected}
        highlights={highlights}
        search={search}
        onSearch={setSearch}
        onToggleClass={toggleClass}
        onClearClasses={() => setSelected([])}
        options={options}
        onToggleOption={toggleOption}
      />
    </div>
  )
}

/* ── Pieces ───────────────────────────────────────────────────────────── */

function Stats({ manifest, outlierMode }) {
  return (
    <div className="pointer-events-none absolute left-4 top-4 rounded-input border border-canvas-hairline bg-canvas-panel px-3 py-2 text-xs text-canvas-muted backdrop-blur-sm">
      <p className="font-medium text-canvas-ink">{manifest.name}</p>
      <p className="mt-0.5">
        {manifest.count.toLocaleString()} images · {manifest.classes.length} classes
      </p>
      <p className="mt-0.5">
        {outlierMode
          ? 'Colour = distance from own class centroid'
          : `3 components hold ${(manifest.projection.totalExplained * 100).toFixed(0)}% of variance`}
      </p>
    </div>
  )
}

function HoverCard({ bundle, index, at }) {
  const { manifest, items, atlas } = bundle
  const names = items.classes[index].map((id) => manifest.classes[id]?.name).filter(Boolean)
  const distance = items.distance[index]

  // Flip the card across the cursor near the edges so it never leaves the canvas.
  const flipX = at.x > at.w - 260
  const flipY = at.y > at.h - 170

  return (
    <div
      className="pointer-events-none absolute z-10 w-max max-w-[240px] rounded-card border border-canvas-hairline bg-canvas-panel p-2 shadow-card-hover backdrop-blur-sm"
      style={{
        left: at.x + (flipX ? -12 : 12),
        top: at.y + (flipY ? -12 : 12),
        transform: `translate(${flipX ? '-100%' : '0'}, ${flipY ? '-100%' : '0'})`,
      }}
    >
      {atlas && (
        <div
          className="rounded-input bg-white/5 bg-no-repeat"
          style={atlas.styleFor(index, 128)}
          aria-hidden="true"
        />
      )}
      <p className="mt-2 max-w-[128px] truncate text-xs font-medium text-canvas-ink">
        {items.name[index]}
      </p>
      {names.length > 0 && <p className="text-xs text-canvas-muted">{names.join(', ')}</p>}
      {distance !== null && (
        <p className="text-xs text-canvas-muted">
          {distance.toFixed(2)} from centroid
          {distance >= manifest.outlierThreshold && (
            <span className="ml-1 font-medium text-canvas-ink">· outlier</span>
          )}
        </p>
      )}
    </div>
  )
}

const TOGGLES = [
  { key: 'showCentroids', label: 'Centroids' },
  { key: 'showHulls', label: 'Hulls' },
  { key: 'showAxes', label: 'Variance axis' },
  { key: 'outlierMode', label: 'Outliers' },
  { key: 'spinning', label: 'Spin' },
]

/**
 * The controls, in a column beside the canvas.
 *
 * A sidebar rather than a strip underneath because the class list is the part
 * that needs room: wrapped chips in a 96px-tall box meant scrolling two rows at
 * a time through a 37-breed dataset. One row per class, scrolling the full
 * height of the window, is the whole reason this moved.
 */
function Sidebar({
  loading,
  classes,
  allClasses,
  selected,
  highlights,
  search,
  onSearch,
  onToggleClass,
  onClearClasses,
  options,
  onToggleOption,
}) {
  return (
    <aside className="flex w-[22rem] shrink-0 flex-col border-l border-hairline bg-surface">
      <section className="shrink-0 border-b border-hairline px-4 py-3">
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
          Overlays
        </h2>
        <div className="grid grid-cols-2 gap-1.5">
          {TOGGLES.map((toggle) => (
            <button
              key={toggle.key}
              type="button"
              onClick={() => onToggleOption(toggle.key)}
              aria-pressed={options[toggle.key]}
              className={
                options[toggle.key]
                  ? 'inline-flex h-8 items-center justify-center rounded-pill border border-brand-200 bg-brand-50 px-3 text-xs font-medium text-brand-700'
                  : 'inline-flex h-8 items-center justify-center rounded-pill border border-hairline bg-field px-3 text-xs font-medium text-body hover:border-brand-200'
              }
            >
              {toggle.label}
            </button>
          ))}
        </div>
      </section>

      <section className="flex min-h-0 flex-1 flex-col px-4 py-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">
            Classes
          </h2>
          {selected.length > 0 && !options.outlierMode && (
            <button
              type="button"
              onClick={onClearClasses}
              className="text-xs font-medium text-brand-600 hover:text-brand-700"
            >
              Clear
            </button>
          )}
        </div>

        {options.outlierMode ? (
          <p className="text-xs text-muted">
            Outlier mode colours every point by distance from its own centroid, so
            per-class colours are turned off while it is on.
          </p>
        ) : loading ? (
          <p className="text-xs text-muted">Loading…</p>
        ) : (
          <>
            <input
              type="search"
              value={search}
              onChange={(event) => onSearch(event.target.value)}
              placeholder={`Filter ${allClasses.length} classes…`}
              className="h-8 w-full rounded-input border border-hairline bg-field px-2.5 text-xs text-ink placeholder:text-muted"
            />
            <p className="mt-2 text-xs text-muted">
              {selected.length
                ? `${selected.length} of ${MAX_HIGHLIGHTS} highlighted`
                : 'Pick a class to colour it'}
            </p>

            <div className="-mx-1 mt-1 min-h-0 flex-1 overflow-y-auto px-1">
              {classes.map((entry) => {
                const color = highlights.get(entry.id)
                return (
                  <button
                    key={entry.id}
                    type="button"
                    onClick={() => onToggleClass(entry.id)}
                    aria-pressed={Boolean(color)}
                    className="flex w-full items-center gap-2 rounded-input px-2 py-1.5 text-left text-xs font-medium text-body hover:bg-surface-subtle"
                    style={color ? { background: `${color}1f`, color } : undefined}
                  >
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-pill"
                      style={{ background: color ?? '#b1c4db' }}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                    <span className="shrink-0 tabular-nums text-muted">{entry.count}</span>
                  </button>
                )
              })}
              {classes.length === 0 && (
                <p className="px-2 py-1.5 text-xs text-muted">
                  No class matches “{search}”.
                </p>
              )}
            </div>
          </>
        )}
      </section>
    </aside>
  )
}

/**
 * The images at either end of a class's axis of maximum variance — what the
 * cluster's longest dimension actually corresponds to, shown rather than named.
 */
function EigenStrip({ bundle, entry, color }) {
  const { atlas, items } = bundle
  if (!atlas) return null

  const row = (indices, label) => (
    <div className="flex items-center gap-2">
      <span className="w-16 shrink-0 text-right text-xs text-muted">{label}</span>
      <div className="flex gap-1.5 overflow-x-auto">
        {indices.map((index) => (
          <div
            key={index}
            className="shrink-0 rounded-input border border-hairline bg-surface-subtle bg-no-repeat"
            style={atlas.styleFor(index, 56)}
            title={items.name[index]}
          />
        ))}
      </div>
    </div>
  )

  return (
    <div className="shrink-0 border-t border-hairline bg-surface px-4 py-3">
      <p className="mb-2 text-xs text-muted">
        <span className="font-medium" style={{ color }}>
          {entry.name}
        </span>{' '}
        along its axis of maximum variance — the two ends of what this class varies in.
      </p>
      <div className="space-y-1.5">
        {row(entry.eigenImages.low, 'one end')}
        {row(entry.eigenImages.high, 'other end')}
      </div>
    </div>
  )
}
