import { createServer } from "vite";

process.env.VITE_API_BASE_URL ||= "http://127.0.0.1:8001";

const server = await createServer({
  configFile: "./vite.config.js",
  clearScreen: false,
  optimizeDeps: {
    force: true,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
});

await server.listen();
server.printUrls();

async function shutdown() {
  await server.close();
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

setInterval(() => {}, 1 << 30);
