/**
 * The page shell — what `latent-explorer view` serves.
 *
 * Fills the window rather than centring a fixed-width card: this is a desktop
 * tool with a whole monitor to work on, and the canvas should get it. The
 * header is deliberately one compact row for the same reason.
 */

import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import Viewer from './Viewer.jsx'
import './index.css'

function App() {
  const [bundles, setBundles] = useState(null)
  const [active, setActive] = useState(null)
  const [error, setError] = useState(null)

  // serve.py writes this next to the symlinked bundle directories.
  useEffect(() => {
    fetch('bundles/index.json')
      .then((response) => {
        if (!response.ok) throw new Error('No bundles are being served.')
        return response.json()
      })
      .then((list) => {
        setBundles(list)
        setActive(list[0] ?? null)
      })
      .catch((err) => setError(err.message))
  }, [])

  return (
    <div className="flex h-dvh flex-col bg-surface">
      <header className="flex shrink-0 items-center gap-4 border-b border-hairline bg-surface px-5 py-2.5">
        <div className="min-w-0">
          <h1 className="text-sm leading-tight">Latent Space Explorer</h1>
          <p className="truncate text-xs text-muted">
            CLIP embeddings projected to three dimensions. Visual similarity is distance.
          </p>
        </div>

        {bundles && bundles.length > 1 && (
          <select
            value={active?.id ?? ''}
            onChange={(event) =>
              setActive(bundles.find((b) => b.id === event.target.value))
            }
            className="ml-auto h-8 shrink-0 rounded-input border border-hairline bg-field px-2.5 text-xs text-ink"
          >
            {bundles.map((bundle) => (
              <option key={bundle.id} value={bundle.id}>
                {bundle.name} ({bundle.count})
              </option>
            ))}
          </select>
        )}
      </header>

      {error ? (
        <div className="grid flex-1 place-items-center px-5">
          <div className="max-w-md text-center">
            <p className="font-medium text-ink">{error}</p>
            <p className="mt-2 text-sm text-body">
              Build one first:{' '}
              <code className="rounded-input bg-field px-1.5 py-0.5 text-xs">
                latent-explorer build datasets/your-set -o bundles/your-set
              </code>
            </p>
          </div>
        </div>
      ) : (
        active && <Viewer bundleUrl={active.path} />
      )}
    </div>
  )
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
