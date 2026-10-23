import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Builds the viewer straight into the Python package, so an installed copy of
// latent-explorer can serve it with no Node present. The output under
// src/latent_explorer/viewer/ is committed for that reason — rebuild it and
// commit the result whenever anything in web/src/ changes.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Served from the filesystem root of a plain http.server, so keep URLs relative.
  base: './',
  // `npm run dev` has no bundles of its own to serve. Borrow them from a
  // `latent-explorer view` running alongside it, so hot reload works against
  // real data instead of the viewer's empty state.
  server: {
    proxy: {
      '/bundles': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: '../src/latent_explorer/viewer',
    emptyOutDir: true,
    sourcemap: false,
  },
})
