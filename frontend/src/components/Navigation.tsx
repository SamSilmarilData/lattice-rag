import React, { useEffect, useState } from 'react';
import { Database, Cpu, Activity, UploadCloud, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import { HealthResponse } from '../types';

interface NavigationProps {
  activeTab: 'studio' | 'eval';
  setActiveTab: (tab: 'studio' | 'eval') => void;
  onOpenIngest: () => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  setActiveTab,
  onOpenIngest,
}) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (err) {
      console.error('Failed to fetch health:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-border bg-card/80 backdrop-blur px-6 py-3 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo & Product Title */}
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-white tracking-tight text-base">lattice-rag</span>
              <span className="px-1.5 py-0.5 text-[10px] font-mono font-medium uppercase bg-cyan-950/60 text-cyan-400 border border-cyan-800/40 rounded">
                v0.6.0
              </span>
            </div>
            <p className="text-[11px] text-gray-400 hidden sm:block">
              Zero-Cloud-Cost Hybrid GraphRAG Studio
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex items-center bg-background/80 p-1 rounded-lg border border-border">
          <button
            onClick={() => setActiveTab('studio')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'studio'
                ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Engineering Studio
          </button>
          <button
            onClick={() => setActiveTab('eval')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'eval'
                ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Evaluation Matrix
          </button>
        </nav>

        {/* Health Indicators & Actions */}
        <div className="flex items-center space-x-3">
          {/* Service Health Pills */}
          <div className="hidden lg:flex items-center space-x-2 text-[11px] text-gray-400 bg-background/60 px-2.5 py-1 rounded-md border border-border/80">
            <div className="flex items-center space-x-1" title="LatticeDB In-Process Property Graph">
              <Database className="w-3 h-3 text-cyan-400" />
              <span>DB</span>
              {health?.latticedbConnected ? (
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
              ) : (
                <AlertCircle className="w-3 h-3 text-gray-500" />
              )}
            </div>

            <span className="text-gray-600">|</span>

            <div className="flex items-center space-x-1" title="TypeSafe Jev System One Router">
              <Cpu className="w-3 h-3 text-indigo-400" />
              <span>Jev</span>
              {health?.typesafeConfigured ? (
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
              ) : (
                <AlertCircle className="w-3 h-3 text-amber-500" />
              )}
            </div>

            <span className="text-gray-600">|</span>

            <div className="flex items-center space-x-1" title="Groq Llama-3.3-70B Synthesizer">
              <span className="text-orange-400 font-mono text-[10px]">Groq</span>
              {health?.groqConfigured ? (
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
              ) : (
                <AlertCircle className="w-3 h-3 text-amber-500" />
              )}
            </div>

            <button
              onClick={fetchHealth}
              disabled={loading}
              className="text-gray-500 hover:text-gray-300 ml-1"
              title="Refresh health status"
            >
              <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* Ingest Document Trigger */}
          <button
            onClick={onOpenIngest}
            className="flex items-center space-x-1.5 px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-semibold text-xs rounded-md shadow-sm transition"
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Ingest Document</span>
          </button>
        </div>
      </div>
    </header>
  );
};
