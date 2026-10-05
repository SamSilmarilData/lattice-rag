import React, { useState } from 'react';
import { Navigation } from './components/Navigation';
import { QueryStudio } from './components/QueryStudio';
import { GraphExplorer } from './components/GraphExplorer';
import { EvalMatrix } from './components/EvalMatrix';
import { IngestModal } from './components/IngestModal';
import { SubgraphDTO } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'studio' | 'eval'>('studio');
  const [isIngestOpen, setIsIngestOpen] = useState(false);
  const [traversedPath, setTraversedPath] = useState<SubgraphDTO | null>(null);
  const [presetQuery, setPresetQuery] = useState<string | null>(null);
  const [graphRefreshKey, setGraphRefreshKey] = useState(0);

  const handleTestQueryInStudio = (query: string) => {
    setPresetQuery(query);
    setActiveTab('studio');
  };

  const handleIngestSuccess = () => {
    // Trigger graph refresh
    setGraphRefreshKey((k) => k + 1);
  };

  return (
    <div className="min-h-screen bg-background text-gray-100 flex flex-col font-sans">
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenIngest={() => setIsIngestOpen(true)}
      />

      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 sm:p-6 overflow-hidden flex flex-col">
        {activeTab === 'studio' ? (
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-[750px]">
            {/* Left Pane: Query Studio & Telemetry */}
            <div className="h-full overflow-y-auto pr-1">
              <QueryStudio
                onGraphUpdate={setTraversedPath}
                presetQuery={presetQuery}
                onClearPreset={() => setPresetQuery(null)}
              />
            </div>

            {/* Right Pane: 2D Interactive Knowledge Graph Explorer */}
            <div className="h-[500px] lg:h-full">
              <GraphExplorer
                key={graphRefreshKey}
                traversedPath={traversedPath}
              />
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <EvalMatrix onTestQueryInStudio={handleTestQueryInStudio} />
          </div>
        )}
      </main>

      <IngestModal
        isOpen={isIngestOpen}
        onClose={() => setIsIngestOpen(false)}
        onIngestSuccess={handleIngestSuccess}
      />
    </div>
  );
};

export default App;
