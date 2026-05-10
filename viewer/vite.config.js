import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import fs from 'node:fs';

const repoRoot = path.resolve(__dirname, '..');
const docxSource = path.join(repoRoot, 'диплом_ред_с_приложениями.docx');

function watchDocxPlugin() {
  return {
    name: 'watch-source-docx',
    configureServer(server) {
      if (!fs.existsSync(docxSource)) {
        server.config.logger.warn(
          `[viewer] source docx not found: ${docxSource}`
        );
        return;
      }
      // python-docx и Word сохраняют через rename — обычный watcher теряет inode.
      // fs.watchFile с интервалом 500мс надёжнее: видит и rename, и truncate.
      let lastReload = 0;
      fs.watchFile(docxSource, { interval: 500 }, (cur, prev) => {
        if (cur.mtimeMs === prev.mtimeMs && cur.size === prev.size) return;
        const now = Date.now();
        if (now - lastReload < 800) return;
        lastReload = now;
        server.config.logger.info(
          `[viewer] docx changed, triggering full reload`,
          { timestamp: true }
        );
        server.ws.send({ type: 'full-reload', path: '*' });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), watchDocxPlugin()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    fs: {
      allow: [path.resolve(__dirname), repoRoot],
    },
  },
});
