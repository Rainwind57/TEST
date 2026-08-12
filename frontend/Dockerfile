# 前端镜像（P2-11）：多阶段构建，产物用 nginx 托管
FROM node:20-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --registry https://registry.npmmirror.com
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# 代理 /api 到后端容器（前端 SPA 访问 API 免跨域）
RUN echo 'server { \
    listen 5173; \
    location / { root /usr/share/nginx/html; try_files $uri $uri/ /index.html; } \
    location /api/ { proxy_pass http://backend:8000; } \
}' > /etc/nginx/conf.d/default.conf
EXPOSE 5173
CMD ["nginx", "-g", "daemon off;"]
