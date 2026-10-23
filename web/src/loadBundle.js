/**
 * Reads a bundle written by `latent-explorer build`.
 *
 * The format is documented in docs/bundle-format.md. Three files: a manifest of
 * everything derived (classes, centroids, hulls, eigen axes), a raw float32
 * blob of positions, and a columnar item table. Positions are binary because
 * they go straight into a GPU buffer — parsing 1500 points out of JSON to then
 * copy them into a Float32Array is work for nothing.
 */

const SUPPORTED_VERSION = 1

export async function loadBundle(baseUrl, { signal } = {}) {
  const base = baseUrl.replace(/\/$/, '')

  const manifest = await fetchJson(`${base}/manifest.json`, signal)
  if (manifest.version !== SUPPORTED_VERSION) {
    throw new Error(
      `Bundle format v${manifest.version} but this viewer reads v${SUPPORTED_VERSION}. Rebuild it.`,
    )
  }

  const [positionBuffer, items] = await Promise.all([
    fetchBinary(`${base}/${manifest.points.file}`, signal),
    fetchJson(`${base}/${manifest.items.file}`, signal),
  ])

  const positions = new Float32Array(positionBuffer)
  const expected = manifest.count * manifest.points.stride
  if (positions.length !== expected) {
    throw new Error(
      `Bundle is inconsistent: ${positions.length} coordinates for ${manifest.count} points.`,
    )
  }

  return { manifest, positions, items, base, atlas: atlasHelper(manifest, base) }
}

async function fetchJson(url, signal) {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`)
  return response.json()
}

async function fetchBinary(url, signal) {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`)
  return response.arrayBuffer()
}

/**
 * Maps a point index to its tile in the thumbnail atlas.
 *
 * Tiles are laid out in point order, so this is pure arithmetic — there is no
 * lookup table in the bundle, and none is needed.
 */
function atlasHelper(manifest, base) {
  const atlas = manifest.thumbnails
  if (!atlas) return null

  const perPage = atlas.cols * atlas.rows

  return {
    tile: atlas.tile,
    cols: atlas.cols,
    rows: atlas.rows,
    /** CSS background properties that crop the sheet down to one tile. */
    styleFor(index, displaySize = atlas.tile * 2) {
      const page = Math.floor(index / perPage)
      const slot = index % perPage
      const scale = displaySize / atlas.tile
      return {
        width: `${displaySize}px`,
        height: `${displaySize}px`,
        backgroundImage: `url(${base}/${atlas.pages[page]})`,
        backgroundPosition: `-${(slot % atlas.cols) * displaySize}px -${
          Math.floor(slot / atlas.cols) * displaySize
        }px`,
        backgroundSize: `${atlas.cols * atlas.tile * scale}px ${
          atlas.rows * atlas.tile * scale
        }px`,
        imageRendering: 'auto',
      }
    },
  }
}
