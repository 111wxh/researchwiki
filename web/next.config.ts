import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 供 Docker 打包：产出 .next/standalone（自包含 server.js，镜像里不用带完整 node_modules）
  output: "standalone",
};

export default nextConfig;
