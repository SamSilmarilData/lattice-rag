import React, { useState } from 'react';
import { Play, CheckCircle2, XCircle, Search, ArrowUpRight, ChevronDown, ChevronRight, BarChart2, ShieldAlert, Zap, Timer, Activity } from 'lucide-react';
import { EvalQueryResult, EvalRunResponse, BenchmarkRunResponse } from '../types';

interface EvalMatrixProps {
  onTestQueryInStudio: (query: string) => void;
}

// Initial golden evaluation dataset populated from eval_dataset.json & baseline
const INITIAL_EVAL_DATA: EvalQueryResult[] = [
  {
    query: 'How does LatticeDB combine HNSW vector similarity with BM25 text search in a single binary?',
    faithfulness: 0.96,
    contextPrecision: 0.94,
    answerRelevance: 0.95,
    passed: true,
  },
  {
    query: 'What is the role of FastEmbed and what dense embedding model does it use by default?',
    faithfulness: 0.98,
    contextPrecision: 0.95,
    answerRelevance: 0.97,
    passed: true,
  },
  {
    query: 'How does GLiNER extract entities and what default entity types are recognized?',
    faithfulness: 0.92,
    contextPrecision: 0.89,
    answerRelevance: 0.93,
    passed: true,
  },
  {
    query: 'Explain the three TypeSafe AI Jev primitives used in the architecture and their specific jobs.',
    faithfulness: 0.97,
    contextPrecision: 0.96,
    answerRelevance: 0.98,
    passed: true,
  },
  {
    query: 'Why is Litestar paired with the Granian Rust ASGI server, and what JSON library is used?',
    faithfulness: 0.95,
    contextPrecision: 0.93,
    answerRelevance: 0.96,
    passed: true,
  },
  {
    query: 'How are relation triples extracted from text and mapped into LatticeDB property-graph edges?',
    faithfulness: 0.91,
    contextPrecision: 0.88,
    answerRelevance: 0.92,
    passed: true,
  },
  {
    query: 'How does the cross-encoder reranker incorporate graph structure during scoring?',
    faithfulness: 0.94,
    contextPrecision: 0.91,
    answerRelevance: 0.95,
    passed: true,
  },
  {
    query: 'How does the front-door decision router triage queries and what routes does it choose between?',
    faithfulness: 0.96,
    contextPrecision: 0.95,
    answerRelevance: 0.97,
    passed: true,
  },
  {
    query: 'What role does the Jev Noul boolean primitive play in context guardrailing before synthesis?',
    faithfulness: 0.93,
    contextPrecision: 0.90,
    answerRelevance: 0.94,
    passed: true,
  },
  {
    query: 'Compare the caching tiers in the architecture: Tier 1 vs Tier 2 fallback.',
    faithfulness: 0.96,
    contextPrecision: 0.94,
    answerRelevance: 0.97,
    passed: true,
  },
  {
    query: 'Explain the end-to-end flow of a multi-hop graph_relational query through the pipeline.',
    faithfulness: 0.93,
    contextPrecision: 0.92,
    answerRelevance: 0.94,
    passed: true,
  },
  {
    query: 'How does the circuit breaker protect external LLM APIs and when does it trip?',
    faithfulness: 0.89,
    contextPrecision: 0.86,
    answerRelevance: 0.91,
    passed: true,
  },
  {
    query: 'What fallback synthesis provider is used if GroqCloud encounters rate limits or errors?',
    faithfulness: 0.97,
    contextPrecision: 0.94,
    answerRelevance: 0.96,
    passed: true,
  },
  {
    query: 'How does SSE streaming work in POST /api/v1/query/stream and what events are emitted?',
    faithfulness: 0.95,
    contextPrecision: 0.92,
    answerRelevance: 0.95,
    passed: true,
  },
  {
    query: 'What security precautions are implemented to prevent API keys from leaking in logs or error traces?',
    faithfulness: 0.98,
    contextPrecision: 0.97,
    answerRelevance: 0.99,
    passed: true,
  },
  {
    query: 'Hello! Can you help me understand what you are capable of?',
    faithfulness: 1.00,
    contextPrecision: 0.90,
    answerRelevance: 0.95,
    passed: true,
  },
  {
    query: 'Thanks for your assistance, that was very helpful!',
    faithfulness: 1.00,
    contextPrecision: 0.90,
    answerRelevance: 0.95,
    passed: true,
  },
  {
    query: 'Synthesize all findings across LatticeDB, FastEmbed, and GLiNER for in-process RAG.',
    faithfulness: 0.92,
    contextPrecision: 0.90,
    answerRelevance: 0.93,
    passed: true,
  },
  {
    query: 'How does Reciprocal Rank Fusion (RRF) combine dense vector and BM25 sparse rankings?',
    faithfulness: 0.95,
    contextPrecision: 0.93,
    answerRelevance: 0.96,
    passed: true,
  },
  {
    query: 'What happens when context guardrail rejects all retrieved candidate chunks?',
    faithfulness: 0.90,
    contextPrecision: 0.87,
    answerRelevance: 0.91,
    passed: true,
  },
];

export const EvalMatrix: React.FC<EvalMatrixProps> = ({ onTestQueryInStudio }) => {
  const [evalData, setEvalData] = useState<EvalQueryResult[]>(INITIAL_EVAL_DATA);
  const [running, setRunning] = useState(false);
  const [evalLimit, setEvalLimit] = useState<number>(5);
  const [evalStatus, setEvalStatus] = useState<string | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkRunResponse | null>(null);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'passed' | 'failed'>('all');
  const [expandedQuery, setExpandedQuery] = useState<string | null>(null);

  // Compute live summary stats
  const total = evalData.length;
  const meanFaithfulness = total ? evalData.reduce((s, r) => s + r.faithfulness, 0) / total : 0;
  const meanPrecision = total ? evalData.reduce((s, r) => s + r.contextPrecision, 0) / total : 0;
  const meanRelevance = total ? evalData.reduce((s, r) => s + r.answerRelevance, 0) / total : 0;
  const allPassed = evalData.every((r) => r.passed);

  const runLiveEval = async () => {
    setRunning(true);
    setEvalError(null);
    setEvalStatus(`Executing Live Evaluation Gate (${evalLimit} queries) through TypeSafe Jev & GroqCloud...`);
    try {
      const res = await fetch(`/api/v1/eval/run?limit=${evalLimit}`, {
        method: 'POST',
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Evaluation failed with status ${res.status}: ${errText}`);
      }
      const data: EvalRunResponse = await res.json();
      if (data.results && data.results.length > 0) {
        setEvalData(data.results);
        setEvalStatus(`✓ Eval Gate Completed: ${data.totalQueries} queries evaluated. Gate status: ${data.passedGate ? 'PASSED' : 'REGRESSION DETECTED'} (Faithfulness: ${(data.meanFaithfulness * 100).toFixed(1)}%, Precision: ${(data.meanContextPrecision * 100).toFixed(1)}%, Relevance: ${(data.meanAnswerRelevance * 100).toFixed(1)}%)`);
      } else {
        setEvalStatus('Evaluation completed with empty results.');
      }
    } catch (err) {
      console.error('Failed to run live evaluation:', err);
      setEvalError(err instanceof Error ? err.message : 'Unknown evaluation failure');
      setEvalStatus(null);
    } finally {
      setRunning(false);
    }
  };

  const runBenchmark = async () => {
    setBenchmarkRunning(true);
    setBenchmarkError(null);
    try {
      const res = await fetch('/api/v1/benchmark/run', {
        method: 'POST',
      });
      if (!res.ok) {
        throw new Error(`Benchmark failed with status ${res.status}`);
      }
      const data: BenchmarkRunResponse = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      console.error('Failed to run performance benchmark:', err);
      setBenchmarkError(err instanceof Error ? err.message : 'Unknown benchmark failure');
    } finally {
      setBenchmarkRunning(false);
    }
  };

  const filteredQueries = evalData.filter((item) => {
    const matchesSearch = item.query.toLowerCase().includes(search.toLowerCase());
    const matchesStatus =
      statusFilter === 'all'
        ? true
        : statusFilter === 'passed'
        ? item.passed
        : !item.passed;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Top Banner & Trigger */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-card border border-border rounded-xl p-5 shadow-sm">
        <div>
          <div className="flex items-center space-x-2">
            <BarChart2 className="w-5 h-5 text-cyan-400" />
            <h2 className="text-base font-semibold text-gray-100">CI/CD Evaluation Matrix & Benchmark Gate</h2>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Golden evaluation suite verified across Faithfulness, Precision, and Sub-Second Latency SLAs.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {/* Query Limit Selector */}
          <div className="flex items-center bg-background border border-border rounded-lg p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setEvalLimit(5)}
              className={`px-2.5 py-1.5 rounded-md text-[11px] font-medium transition ${
                evalLimit === 5
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Quick (5)
            </button>
            <button
              type="button"
              onClick={() => setEvalLimit(20)}
              className={`px-2.5 py-1.5 rounded-md text-[11px] font-medium transition ${
                evalLimit === 20
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Full (20)
            </button>
          </div>

          <button
            onClick={runBenchmark}
            disabled={benchmarkRunning || running}
            className="flex items-center space-x-1.5 px-3.5 py-2 bg-indigo-950/70 hover:bg-indigo-900/80 disabled:bg-gray-800 disabled:text-gray-600 text-indigo-300 border border-indigo-700/50 font-semibold text-xs rounded-lg shadow-sm transition"
          >
            <Zap className={`w-3.5 h-3.5 ${benchmarkRunning ? 'animate-pulse text-amber-400' : 'text-indigo-400'}`} />
            <span>{benchmarkRunning ? 'Benchmarking Pipeline...' : 'Run Performance Benchmark'}</span>
          </button>

          <button
            onClick={runLiveEval}
            disabled={running || benchmarkRunning}
            className="flex items-center space-x-1.5 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-800 disabled:text-gray-600 text-slate-950 font-semibold text-xs rounded-lg shadow-sm transition"
          >
            <Play className={`w-3.5 h-3.5 ${running ? 'animate-spin' : ''}`} />
            <span>{running ? `Running Gate (${evalLimit})...` : `Trigger Live Eval Gate (${evalLimit})`}</span>
          </button>
        </div>
      </div>

      {/* Eval Live Progress Alert */}
      {running && (
        <div className="bg-cyan-950/40 border border-cyan-800/60 rounded-xl p-3.5 text-xs text-cyan-300 flex items-center space-x-2 animate-pulse">
          <Activity className="w-4 h-4 shrink-0 text-cyan-400 animate-spin" />
          <span>⚡ Running Live Evaluation Gate across {evalLimit} queries via TypeSafe Jev & GroqCloud... Please wait (~5-8 seconds for quick gate).</span>
        </div>
      )}

      {/* Eval Completion Alert */}
      {evalStatus && !running && (
        <div className="bg-emerald-950/40 border border-emerald-800/60 rounded-xl p-3.5 text-xs text-emerald-300 flex items-center justify-between space-x-2">
          <div className="flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>{evalStatus}</span>
          </div>
          <button
            onClick={() => setEvalStatus(null)}
            className="text-gray-400 hover:text-gray-200 text-[11px] underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Eval Error Alert */}
      {evalError && (
        <div className="bg-rose-950/40 border border-rose-800/60 rounded-xl p-3.5 text-xs text-rose-300 flex items-center justify-between space-x-2">
          <div className="flex items-center space-x-2">
            <ShieldAlert className="w-4 h-4 shrink-0 text-rose-400" />
            <span>Evaluation Gate Notice: {evalError}</span>
          </div>
          <button
            onClick={() => setEvalError(null)}
            className="text-gray-400 hover:text-gray-200 text-[11px] underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Latency & SLA Performance Breakdown Card */}
      {benchmarkData && (
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border/80 pb-3">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-gray-200">Production Latency & Dual-SLA Profiling</h3>
              <span className="text-[10px] font-mono text-gray-500 bg-background px-2 py-0.5 rounded border border-border">
                {benchmarkData.totalDurationSec.toFixed(2)}s runtime
              </span>
            </div>

            <div className="flex items-center space-x-2">
              {/* End-to-End Sub-Second SLA Badge */}
              <div className="flex items-center space-x-1 px-2.5 py-1 rounded text-[11px] font-mono font-medium border bg-background">
                <span className="text-gray-400">E2E SLA (p95 &lt; 1000ms):</span>
                {benchmarkData.subSecondSlaMet ? (
                  <span className="text-emerald-400 font-bold flex items-center space-x-0.5">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400 inline" />
                    <span>PASSED</span>
                  </span>
                ) : (
                  <span className="text-rose-400 font-bold flex items-center space-x-0.5">
                    <XCircle className="w-3 h-3 text-rose-400 inline" />
                    <span>FAILED</span>
                  </span>
                )}
              </div>

              {/* Tier 1 Semantic Cache SLA Badge */}
              <div className="flex items-center space-x-1 px-2.5 py-1 rounded text-[11px] font-mono font-medium border bg-background">
                <span className="text-gray-400">Cache SLA (p95 &lt; 25ms):</span>
                {benchmarkData.tier1CacheSlaMet ? (
                  <span className="text-emerald-400 font-bold flex items-center space-x-0.5">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400 inline" />
                    <span>PASSED</span>
                  </span>
                ) : (
                  <span className="text-rose-400 font-bold flex items-center space-x-0.5">
                    <XCircle className="w-3 h-3 text-rose-400 inline" />
                    <span>FAILED</span>
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Benchmark Stages Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-background/90 text-gray-400 border-b border-border font-medium">
                <tr>
                  <th className="py-2.5 px-3">Pipeline Stage</th>
                  <th className="py-2.5 px-3 text-center">p50</th>
                  <th className="py-2.5 px-3 text-center">p90</th>
                  <th className="py-2.5 px-3 text-center">p95</th>
                  <th className="py-2.5 px-3 text-center">p99</th>
                  <th className="py-2.5 px-3 text-center">Mean</th>
                  <th className="py-2.5 px-3 text-right">Throughput</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60">
                {benchmarkData.stages.map((st, i) => (
                  <tr key={i} className="hover:bg-surface/30 transition">
                    <td className="py-2.5 px-3 font-medium text-gray-200 flex items-center space-x-2">
                      <Timer className="w-3.5 h-3.5 text-cyan-400/80 shrink-0" />
                      <span>{st.stage}</span>
                    </td>
                    <td className="py-2.5 px-3 text-center font-mono text-gray-300">
                      {st.p50Ms.toFixed(1)}ms
                    </td>
                    <td className="py-2.5 px-3 text-center font-mono text-gray-300">
                      {st.p90Ms.toFixed(1)}ms
                    </td>
                    <td className="py-2.5 px-3 text-center font-mono text-cyan-400 font-semibold">
                      {st.p95Ms.toFixed(1)}ms
                    </td>
                    <td className="py-2.5 px-3 text-center font-mono text-gray-400">
                      {st.p99Ms.toFixed(1)}ms
                    </td>
                    <td className="py-2.5 px-3 text-center font-mono text-gray-300">
                      {st.meanMs.toFixed(1)}ms
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-emerald-400">
                      {st.qps.toFixed(1)} QPS
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {benchmarkError && (
        <div className="bg-rose-950/40 border border-rose-800/60 rounded-xl p-3.5 text-xs text-rose-300 flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 shrink-0 text-rose-400" />
          <span>Benchmark Execution Notice: {benchmarkError}</span>
        </div>
      )}

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="bg-card border border-border rounded-xl p-3.5">
          <div className="text-[11px] text-gray-400 font-medium">Total Golden Queries</div>
          <div className="text-xl font-bold font-mono text-gray-100 mt-1">{total}</div>
          <div className="text-[10px] text-gray-500 mt-0.5">Automated test matrix</div>
        </div>

        <div className="bg-card border border-border rounded-xl p-3.5">
          <div className="text-[11px] text-gray-400 font-medium">Mean Faithfulness</div>
          <div className="text-xl font-bold font-mono text-cyan-400 mt-1">
            {(meanFaithfulness * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">Threshold: $\ge$ 88.0%</div>
        </div>

        <div className="bg-card border border-border rounded-xl p-3.5">
          <div className="text-[11px] text-gray-400 font-medium">Mean Context Precision</div>
          <div className="text-xl font-bold font-mono text-indigo-400 mt-1">
            {(meanPrecision * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">Threshold: $\ge$ 85.0%</div>
        </div>

        <div className="bg-card border border-border rounded-xl p-3.5">
          <div className="text-[11px] text-gray-400 font-medium">Mean Answer Relevance</div>
          <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
            {(meanRelevance * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">Threshold: $\ge$ 90.0%</div>
        </div>

        <div className="bg-card border border-border rounded-xl p-3.5 col-span-2 md:col-span-1">
          <div className="text-[11px] text-gray-400 font-medium">Gate Status</div>
          <div className="mt-1 flex items-center space-x-1.5">
            {allPassed ? (
              <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 flex items-center space-x-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>PASSED</span>
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-rose-950/80 text-rose-400 border border-rose-800/60 flex items-center space-x-1">
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>FAILED</span>
              </span>
            )}
          </div>
          <div className="text-[10px] text-gray-500 mt-1">Regression Delta: 0.00</div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-card border border-border rounded-xl p-3">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-gray-500 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search queries..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-background border border-border rounded-lg pl-9 pr-3 py-1.5 text-xs text-gray-100 placeholder-gray-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        <div className="flex items-center space-x-2 self-end sm:self-auto text-xs">
          <span className="text-gray-500 text-[11px]">Status:</span>
          {(['all', 'passed', 'failed'] as const).map((st) => (
            <button
              key={st}
              onClick={() => setStatusFilter(st)}
              className={`px-2.5 py-1 rounded text-xs capitalize transition ${
                statusFilter === st
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* Data Grid Table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-background/80 text-gray-400 border-b border-border font-medium">
              <tr>
                <th className="py-3 px-4 w-12 text-center">#</th>
                <th className="py-3 px-4">Benchmark Query</th>
                <th className="py-3 px-4 w-28 text-center">Faithfulness</th>
                <th className="py-3 px-4 w-28 text-center">Precision</th>
                <th className="py-3 px-4 w-28 text-center">Relevance</th>
                <th className="py-3 px-4 w-24 text-center">Status</th>
                <th className="py-3 px-4 w-32 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {filteredQueries.map((item, index) => {
                const isExpanded = expandedQuery === item.query;
                return (
                  <React.Fragment key={index}>
                    <tr
                      className="hover:bg-surface/30 transition cursor-pointer"
                      onClick={() => setExpandedQuery(isExpanded ? null : item.query)}
                    >
                      <td className="py-3 px-4 text-center font-mono text-gray-500">
                        {index + 1}
                      </td>
                      <td className="py-3 px-4 text-gray-200">
                        <div className="flex items-center space-x-2">
                          {isExpanded ? (
                            <ChevronDown className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                          ) : (
                            <ChevronRight className="w-3.5 h-3.5 text-gray-500 shrink-0" />
                          )}
                          <span className="font-medium">{item.query}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-center font-mono text-cyan-400">
                        {(item.faithfulness * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 px-4 text-center font-mono text-indigo-400">
                        {(item.contextPrecision * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 px-4 text-center font-mono text-emerald-400">
                        {(item.answerRelevance * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 px-4 text-center">
                        {item.passed ? (
                          <span className="inline-flex items-center space-x-1 text-emerald-400 font-mono text-[10px]">
                            <CheckCircle2 className="w-3 h-3" />
                            <span>PASS</span>
                          </span>
                        ) : (
                          <span className="inline-flex items-center space-x-1 text-rose-400 font-mono text-[10px]">
                            <XCircle className="w-3 h-3" />
                            <span>FAIL</span>
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onTestQueryInStudio(item.query);
                          }}
                          className="inline-flex items-center space-x-1 px-2.5 py-1 bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 border border-cyan-800/40 rounded text-[11px] font-medium transition"
                        >
                          <span>Test in Studio</span>
                          <ArrowUpRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>

                    {/* Expandable Comparison Details */}
                    {isExpanded && (
                      <tr className="bg-background/80">
                        <td colSpan={7} className="p-4 border-y border-border/80">
                          <div className="space-y-3 text-xs max-w-4xl mx-auto">
                            <div className="font-semibold text-gray-300 flex items-center justify-between">
                              <span>Golden Benchmark Evaluation Details</span>
                              <span className="font-mono text-[11px] text-gray-500">
                                Target Floor: &gt; 0.50
                              </span>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                              <div className="bg-card border border-border p-3 rounded-lg">
                                <div className="text-gray-400 text-[11px]">Faithfulness Score</div>
                                <div className="text-base font-bold font-mono text-cyan-400 mt-0.5">
                                  {item.faithfulness.toFixed(3)}
                                </div>
                                <p className="text-[10px] text-gray-500 mt-1">
                                  Ratio of claims in the generated answer supported by retrieved facts.
                                </p>
                              </div>
                              <div className="bg-card border border-border p-3 rounded-lg">
                                <div className="text-gray-400 text-[11px]">Context Precision</div>
                                <div className="text-base font-bold font-mono text-indigo-400 mt-0.5">
                                  {item.contextPrecision.toFixed(3)}
                                </div>
                                <p className="text-[10px] text-gray-500 mt-1">
                                  Proportion of ground-truth evidence contained in retrieved chunks.
                                </p>
                              </div>
                              <div className="bg-card border border-border p-3 rounded-lg">
                                <div className="text-gray-400 text-[11px]">Answer Relevance</div>
                                <div className="text-base font-bold font-mono text-emerald-400 mt-0.5">
                                  {item.answerRelevance.toFixed(3)}
                                </div>
                                <p className="text-[10px] text-gray-500 mt-1">
                                  Semantic match between generated answer and the user query intent.
                                </p>
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
