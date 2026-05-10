import { useMemo } from 'react';
import { SuperDocEditor } from '@superdoc-dev/react';
import '@superdoc-dev/react/style.css';

const DOC_PATH = '/diplom.docx';

export default function App() {
  const docUrl = useMemo(() => `${DOC_PATH}?v=${Date.now()}`, []);

  return (
    <div className="viewer-shell">
      <header className="viewer-header">
        <span className="viewer-title">диплом_ред_с_приложениями.docx</span>
        <span className="viewer-mode">read-only</span>
      </header>
      <main className="viewer-main">
        <SuperDocEditor
          document={docUrl}
          documentMode="viewing"
          pagination={true}
          rulers={false}
          onReady={() => {
            console.log('[viewer] superdoc ready');
          }}
        />
      </main>
    </div>
  );
}
