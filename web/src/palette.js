/**
 * Colours for highlighted classes, carried over from the original Python tool.
 *
 * Unhighlighted points stay near-neutral so that selecting a class reads as
 * "this, out of everything" rather than as one more colour competing with the
 * rest. That contrast is the whole reason the base colour is so pale.
 */

export const NEUTRAL = '#aeb7c7'
// Centroids are an annotation, not data, so they get a colour nothing else uses:
// warm and saturated, which separates them from the cool dark canvas and the
// cool pale cloud by hue as well as by brightness. Safe from collisions — a
// selected class's centroid takes that class's colour instead, and centroids
// are hidden outright in outlier mode, where the ramp also runs through amber.
export const CENTROID_NEUTRAL = '#fbbf24'

const PALETTE = [
  '#e74c3c', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd',
  '#8c564b', '#d62728', '#17becf', '#bcbd22', '#7f7f7f',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
  '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
  '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
]

/** Colour for the nth highlighted class, in selection order. */
export function highlightColor(index) {
  return PALETTE[index % PALETTE.length]
}

/** '#rrggbb' → [r, g, b] in 0..1, which is what the point shader wants. */
export function toRgbArray(hex) {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ]
}

/**
 * Outlier ramp: calm blue-grey at the centroid through to alarm red far from it.
 * Sequential rather than categorical, because distance is a magnitude — a
 * categorical palette here would imply groups that don't exist.
 *
 * The low end is light rather than dark: these are drawn on the dark canvas, so
 * a dark "calm" end would sink into the background instead of reading as calm.
 */
const OUTLIER_RAMP = [
  [0.58, 0.65, 0.76],
  [0.38, 0.72, 0.76],
  [0.93, 0.78, 0.40],
  [0.95, 0.36, 0.30],
]

export function outlierColor(t) {
  const clamped = Math.min(1, Math.max(0, t))
  const scaled = clamped * (OUTLIER_RAMP.length - 1)
  const i = Math.min(OUTLIER_RAMP.length - 2, Math.floor(scaled))
  const f = scaled - i
  const a = OUTLIER_RAMP[i]
  const b = OUTLIER_RAMP[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}
