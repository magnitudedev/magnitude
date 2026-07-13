import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    svelte(),
    tailwindcss(),
  ],
  server: {
    port: 4777,
    strictPort: true,
    proxy: {
      '/api': 'http://localhost:4776',
    },
  },
  build: {
    outDir: 'dist',
  },
})
