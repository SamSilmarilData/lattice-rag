import React, { useState } from 'react';
import { X, UploadCloud, CheckCircle2, Sparkles } from 'lucide-react';
import { IngestResponse } from '../types';

interface IngestModalProps {
  isOpen: boolean;
  onClose: () => void;
  onIngestSuccess: () => void;
}

const PRESET_DOCUMENTS = [
  {
    id: 'doc_latticedb_internals',
    title: 'LatticeDB Embedded Property-Graph Architecture',
    text: `LatticeDB is a serverless, embedded property-graph database designed for high-performance in-process execution. Operating similarly to SQLite for graph and vector data, LatticeDB eliminates remote database network boundaries by running entirely inside the host application process. It uniquely unites Cypher multi-hop graph queries with native HNSW vector similarity search using the <=> operator and BM25 inverted index text searches using the @@ operator in a single local binary file. Nodes follow a hierarchical schema comprising Document, Chunk, and Entity labels. Edges include [:HAS_CHUNK] linking documents to chunks, [:CONTAINS] linking chunks to extracted entities, and [:RELATION {type: ...}] linking entities together for multi-hop graph traversals. Because all indices share the same transactional storage layer, hybrid vector-graph queries execute with sub-millisecond latency.`,
  },
  {
    id: 'doc_fastembed_reranking',
    title: 'FastEmbed Quantized Embeddings and Cross-Encoder Reranking',
    text: `FastEmbed provides CPU-quantized dense and sparse vector generation without requiring GPU hardware. The default dense model is BAAI/bge-small-en-v1.5, generating 384-dimensional normalized vector embeddings optimized for cosine similarity. For precision stage-3 reranking, the engine integrates BAAI/bge-reranker-v2-m3 in an INT8 quantized ONNX format from onnx-community/bge-reranker-v2-m3-ONNX. This cross-encoder features an 8,192 token context window, 568 million parameters, and executes inference in under 100 milliseconds on CPU via dynamic token batch padding. During reranking, graph paths traversed in LatticeDB are linearized into structured triples formatted as 'Entity1 --[RELATION]--> Entity2' and appended to candidate text chunks, allowing the cross-encoder to jointly score semantic text relevance and relational graph context.`,
  },
  {
    id: 'doc_gliner_extraction',
    title: 'GLiNER Zero-Shot Named Entity Recognition and Triples',
    text: `The knowledge extraction pipeline employs GLiNER (Generalist and Lightweight Model for Named Entity Recognition). By default, the system leverages urchade/gliner_small-v2.1 as an efficient in-process extractor. GLiNER extracts domain entities corresponding to predefined labels including person, organization, technology, concept, algorithm, database, and protocol. Following entity extraction, the EntityExtractor extracts relation triples using sentence-level co-occurrence heuristics: when two entities co-occur within the same sentence, a semantic directed relationship edge [:RELATION] is created between their respective nodes in LatticeDB with relationship type derived from context. Extracted entities are tied to their originating text chunk through [:CONTAINS] edges, ensuring tight provenance tracking and citation grounding.`,
  },
  {
    id: 'doc_typesafe_system_one',
    title: 'TypeSafe AI Jev System One Decision Architecture',
    text: `TypeSafe AI Jev is a specialized System One decision model trained to provide fast, calibrated judgments rather than generating free-form conversational text. The pipeline utilizes three primary Jev primitives. First, the Choice primitive acts as a front-door router, classifying incoming prompts in 70 to 500 milliseconds into vector_exact, graph_relational, hybrid, chitchat, or massive_context routes. Second, the boolean Noul primitive serves as a context guardrail, evaluating retrieved candidate chunks in parallel and pruning tangential passages with relevance probability below 0.50. Third, the Score primitive evaluates content across ordered descriptive criteria, returning expected-value continuous scores and probability distributions. By decomposing complex workflows into small typed judgments, Jev ensures deterministic control, low token usage, and sub-second pipeline throughput.`,
  },
  {
    id: 'doc_litestar_granian_stack',
    title: 'Litestar and Granian Rust-Optimized Web Framework',
    text: `The web API layer is built on Litestar 2.24+ and served by Granian, a high-performance HTTP server written in Rust that delivers 2 to 4 times the throughput of standard Python Uvicorn servers. All data transfer objects are defined as msgspec.Struct instances with rename='camel' to ensure zero-copy sub-millisecond JSON serialization. The service provides dual query APIs: a synchronous POST /api/v1/query endpoint and a streaming POST /api/v1/query/stream endpoint emitting typed Server-Sent Events (SSE) with stage, token, and done events for frontend telemetry. Error handling complies with RFC 9457 Problem Details, and all logging and error responses pass through automated regex sanitizers to scrub API keys, tokens, and authorization headers before reaching the network.`,
  },
];

export const IngestModal: React.FC<IngestModalProps> = ({ isOpen, onClose, onIngestSuccess }) => {
  const [docId, setDocId] = useState('doc_latticedb_internals');
  const [title, setTitle] = useState('LatticeDB Embedded Property-Graph Architecture');
  const [text, setText] = useState(PRESET_DOCUMENTS[0].text);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSelectPreset = (preset: typeof PRESET_DOCUMENTS[0]) => {
    setDocId(preset.id);
    setTitle(preset.title);
    setText(preset.text);
    setResult(null);
    setError(null);
  };

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || loading) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch('/api/v1/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documentId: docId.trim(),
          title: title.trim(),
          text: text.trim(),
        }),
      });

      if (!res.ok) {
        throw new Error(`Ingest failed with status ${res.status}: ${res.statusText}`);
      }

      const data: IngestResponse = await res.json();
      setResult(data);
      onIngestSuccess();
    } catch (err: unknown) {
      console.error('Ingestion error:', err);
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl animate-scaleIn">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center space-x-2.5">
            <UploadCloud className="w-5 h-5 text-cyan-400" />
            <div>
              <h3 className="text-sm font-semibold text-gray-100">Document Ingestion Pipeline</h3>
              <p className="text-[11px] text-gray-400">
                Chunk text, embed with FastEmbed, extract entities via GLiNER, and link into LatticeDB.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 p-1 rounded-md hover:bg-surface transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* Preset Buttons */}
          <div className="space-y-1.5">
            <div className="flex items-center space-x-1.5 text-xs text-gray-400 font-medium">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>Select Corpus Preset:</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {PRESET_DOCUMENTS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => handleSelectPreset(p)}
                  className={`text-[11px] px-2.5 py-1 rounded-md border transition truncate max-w-[200px] ${
                    docId === p.id
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                      : 'bg-background/80 text-gray-400 border-border hover:bg-surface hover:text-gray-200'
                  }`}
                  title={p.title}
                >
                  {p.title}
                </button>
              ))}
            </div>
          </div>

          <form id="ingest-form" onSubmit={handleIngest} className="space-y-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] text-gray-400 mb-1 font-medium">Document ID</label>
                <input
                  type="text"
                  value={docId}
                  onChange={(e) => setDocId(e.target.value)}
                  className="w-full bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>
              <div>
                <label className="block text-[11px] text-gray-400 mb-1 font-medium">Document Title</label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full bg-background border border-border rounded-lg px-3 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-[11px] text-gray-400 mb-1 font-medium">Document Content</label>
              <textarea
                rows={7}
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="w-full bg-background border border-border rounded-lg p-3 text-xs text-gray-200 font-sans leading-relaxed focus:outline-none focus:border-cyan-500"
                placeholder="Paste technical article or corpus text..."
                required
              />
            </div>
          </form>

          {/* Ingest Result Banner */}
          {result && (
            <div className="bg-emerald-950/40 border border-emerald-800/60 rounded-xl p-4 space-y-2 animate-fadeIn">
              <div className="flex items-center space-x-2 text-emerald-400 font-semibold text-xs">
                <CheckCircle2 className="w-4 h-4" />
                <span>Document Ingested Successfully!</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center pt-1 font-mono text-xs">
                <div className="bg-background/60 p-2 rounded border border-emerald-900/40">
                  <div className="text-gray-400 text-[10px]">Chunks Created</div>
                  <div className="text-emerald-300 font-bold text-sm mt-0.5">{result.chunkCount}</div>
                </div>
                <div className="bg-background/60 p-2 rounded border border-emerald-900/40">
                  <div className="text-gray-400 text-[10px]">Entities Extracted</div>
                  <div className="text-cyan-300 font-bold text-sm mt-0.5">{result.entityCount}</div>
                </div>
                <div className="bg-background/60 p-2 rounded border border-emerald-900/40">
                  <div className="text-gray-400 text-[10px]">Relations Linked</div>
                  <div className="text-indigo-300 font-bold text-sm mt-0.5">{result.relationCount}</div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-rose-950/40 border border-rose-800/60 rounded-xl p-3 text-xs text-rose-300">
              {error}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-3.5 border-t border-border bg-background/40">
          <span className="text-[11px] text-gray-500">
            Powered by FastEmbed &amp; GLiNER (CPU In-Process)
          </span>
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-1.5 text-xs text-gray-400 hover:text-gray-200 transition"
            >
              Close
            </button>
            <button
              type="submit"
              form="ingest-form"
              disabled={loading || !text.trim()}
              className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-800 disabled:text-gray-600 text-slate-950 font-semibold text-xs rounded-lg transition flex items-center space-x-1.5"
            >
              <span>{loading ? 'Ingesting...' : 'Ingest Document'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
