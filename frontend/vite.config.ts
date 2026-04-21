import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/status': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/pipeline': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/video': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/frame-selector': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    }
  }
})
