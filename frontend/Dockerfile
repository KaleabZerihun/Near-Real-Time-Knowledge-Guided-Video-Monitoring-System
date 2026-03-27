# syntax=docker/dockerfile:1

# ─── Stage 1: Build the Vite / React app ─────────────────────────────────────
FROM node:22-slim AS builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
RUN npm run build

# ─── Stage 2: Serve with nginx ────────────────────────────────────────────────
FROM nginx:alpine AS runtime

RUN rm -rf /usr/share/nginx/html/*
COPY --from=builder /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["/bin/sh", "-c", "echo \"\"; echo \"============================================\"; echo \"  Frontend running at: http://localhost:5173\"; echo \"============================================\"; echo \"\"; nginx -g \"daemon off;\""]
