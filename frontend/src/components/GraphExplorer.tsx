import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  Simulation,
  SimulationNodeDatum,
  SimulationLinkDatum,
} from 'd3-force';
import {
  Share2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  X,
  Database,
  Code2,
  Cpu,
  Boxes,
} from 'lucide-react';
import { SubgraphDTO } from '../types';

interface GraphExplorerProps {
  traversedPath: SubgraphDTO | null;
}

interface SimNode extends SimulationNodeDatum {
  id: number;
  name: string;
  label: string;
  x: number;
  y: number;
  vx?: number;
  vy?: number;
}

interface SimEdge extends SimulationLinkDatum<SimNode> {
  sourceId: number;
  targetId: number;
  relationType: string;
  source: SimNode | number;
  target: SimNode | number;
}

const ENTITY_COLORS: Record<string, { bg: string; border: string; text: string; fill: string }> = {
  Technology: { bg: 'bg-cyan-500/20', border: '#06B6D4', text: 'text-cyan-400', fill: '#0891B2' },
  Database: { bg: 'bg-emerald-500/20', border: '#10B981', text: 'text-emerald-400', fill: '#059669' },
  Algorithm: { bg: 'bg-purple-500/20', border: '#A855F7', text: 'text-purple-400', fill: '#9333EA' },
  Protocol: { bg: 'bg-amber-500/20', border: '#F59E0B', text: 'text-amber-400', fill: '#D97706' },
  Concept: { bg: 'bg-rose-500/20', border: '#F43F5E', text: 'text-rose-400', fill: '#E11D48' },
  Library: { bg: 'bg-indigo-500/20', border: '#6366F1', text: 'text-indigo-400', fill: '#4F46E5' },
  Entity: { bg: 'bg-blue-500/20', border: '#3B82F6', text: 'text-blue-400', fill: '#2563EB' },
};

export const GraphExplorer: React.FC<GraphExplorerProps> = ({ traversedPath }) => {
  const [mode, setMode] = useState<'traversed' | 'full'>('full');
  const [fullGraph, setFullGraph] = useState<SubgraphDTO | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<SimNode | null>(null);

  // SVG Pan / Zoom state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isDraggingPan = useRef(false);
  const lastMousePos = useRef({ x: 0, y: 0 });

  // d3 simulation references
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<Simulation<SimNode, SimEdge> | null>(null);
  const [simNodes, setSimNodes] = useState<SimNode[]>([]);
  const [simEdges, setSimEdges] = useState<SimEdge[]>([]);

  // Fetch full knowledge graph on mount
  const fetchFullGraph = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/v1/graph/subgraph?limit=60');
      if (res.ok) {
        const data: SubgraphDTO = await res.json();
        setFullGraph(data);
      }
    } catch (err) {
      console.error('Failed to load full graph:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFullGraph();
  }, [fetchFullGraph]);

  // When a traversed path arrives, default view mode to traversed if there are nodes
  useEffect(() => {
    if (traversedPath && traversedPath.nodes.length > 0) {
      setMode('traversed');
    }
  }, [traversedPath]);

  // Select active dataset based on mode
  const activeData: SubgraphDTO = useMemo(() => {
    if (mode === 'traversed' && traversedPath && traversedPath.nodes.length > 0) {
      return traversedPath;
    }
    return fullGraph || { nodes: [], edges: [] };
  }, [mode, traversedPath, fullGraph]);

  // Initialize and run d3-force simulation
  useEffect(() => {
    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    if (!activeData.nodes || activeData.nodes.length === 0) {
      setSimNodes([]);
      setSimEdges([]);
      return;
    }

    const width = 600;
    const height = 500;

    // Create deep copies for d3 simulation mutation
    const nodes: SimNode[] = activeData.nodes.map((n, i) => ({
      id: n.nodeId,
      name: n.name,
      label: n.label,
      x: width / 2 + (Math.random() - 0.5) * 200,
      y: height / 2 + (Math.random() - 0.5) * 200,
      index: i,
    }));

    const nodeById = new Map<number, SimNode>(nodes.map((n) => [n.id, n]));

    const edges: SimEdge[] = activeData.edges
      .filter((e) => nodeById.has(e.sourceId) && nodeById.has(e.targetId))
      .map((e) => ({
        sourceId: e.sourceId,
        targetId: e.targetId,
        relationType: e.relationType,
        source: nodeById.get(e.sourceId)!,
        target: nodeById.get(e.targetId)!,
      }));

    const sim = forceSimulation<SimNode, SimEdge>(nodes)
      .alphaDecay(0.035)
      .velocityDecay(0.45)
      .force(
        'link',
        forceLink<SimNode, SimEdge>(edges)
          .id((d) => d.id)
          .distance(90)
      )
      .force('charge', forceManyBody().strength(-180))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collision', forceCollide().radius(35))
      .stop();

    // 60-tick synchronous warmup off-screen for instant stabilization
    for (let i = 0; i < 60; ++i) {
      sim.tick();
    }
    setSimNodes([...nodes]);
    setSimEdges([...edges]);

    // Continue remaining simulation settling, throttled by requestAnimationFrame
    let animFrameId: number | null = null;
    sim.restart();
    sim.on('tick', () => {
      if (animFrameId === null) {
        animFrameId = requestAnimationFrame(() => {
          setSimNodes([...nodes]);
          setSimEdges([...edges]);
          animFrameId = null;
        });
      }
    });

    simulationRef.current = sim;

    return () => {
      sim.stop();
      if (animFrameId !== null) {
        cancelAnimationFrame(animFrameId);
      }
    };
  }, [activeData]);

  // Pan and Zoom event handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    isDraggingPan.current = true;
    lastMousePos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingPan.current) return;
    const dx = e.clientX - lastMousePos.current.x;
    const dy = e.clientY - lastMousePos.current.y;
    setPan((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
    lastMousePos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => {
    isDraggingPan.current = false;
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Node selection and connected neighborhood calculation
  const connectedNodeIds = useMemo(() => {
    if (!selectedNode) return null;
    const set = new Set<number>();
    set.add(selectedNode.id);
    activeData.edges.forEach((e) => {
      if (e.sourceId === selectedNode.id) set.add(e.targetId);
      if (e.targetId === selectedNode.id) set.add(e.sourceId);
    });
    return set;
  }, [selectedNode, activeData.edges]);

  const selectedRelations = useMemo(() => {
    if (!selectedNode) return [];
    return activeData.edges.filter(
      (e) => e.sourceId === selectedNode.id || e.targetId === selectedNode.id
    );
  }, [selectedNode, activeData.edges]);

  const getNodeIcon = (label: string) => {
    switch (label) {
      case 'Database':
        return <Database className="w-3 h-3 text-emerald-400" />;
      case 'Technology':
        return <Cpu className="w-3 h-3 text-cyan-400" />;
      case 'Algorithm':
        return <Code2 className="w-3 h-3 text-purple-400" />;
      default:
        return <Boxes className="w-3 h-3 text-indigo-400" />;
    }
  };

  return (
    <div className="flex flex-col h-full bg-card border border-border rounded-xl overflow-hidden relative">
      {/* Top Header & Mode Switcher */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/80 bg-background/50 z-10">
        <div className="flex items-center space-x-2">
          <Share2 className="w-4 h-4 text-cyan-400" />
          <h2 className="text-xs font-semibold text-gray-200">2D Knowledge Graph Visualizer</h2>
          <span className="text-[11px] font-mono text-gray-500">
            ({simNodes.length} nodes, {simEdges.length} edges)
          </span>
        </div>

        {/* Mode Toggle */}
        <div className="flex items-center space-x-1 bg-card/80 p-0.5 rounded-lg border border-border">
          <button
            onClick={() => setMode('traversed')}
            disabled={!traversedPath || traversedPath.nodes.length === 0}
            className={`px-2.5 py-1 text-[11px] font-medium rounded transition ${
              mode === 'traversed'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-gray-400 hover:text-gray-200 disabled:text-gray-600 disabled:hover:text-gray-600'
            }`}
          >
            Query Path {traversedPath && traversedPath.nodes.length > 0 ? `(${traversedPath.nodes.length})` : ''}
          </button>
          <button
            onClick={() => setMode('full')}
            className={`px-2.5 py-1 text-[11px] font-medium rounded transition ${
              mode === 'full'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Full Graph
          </button>
        </div>
      </div>

      {/* Interactive Controls Overlay */}
      <div className="absolute left-4 bottom-4 flex flex-col space-y-1.5 z-10 bg-card/90 border border-border/80 p-1 rounded-lg backdrop-blur shadow-lg">
        <button
          onClick={() => setZoom((z) => Math.min(z * 1.25, 3))}
          className="p-1.5 text-gray-400 hover:text-cyan-400 hover:bg-surface rounded transition"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(z * 0.8, 0.4))}
          className="p-1.5 text-gray-400 hover:text-cyan-400 hover:bg-surface rounded transition"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={resetView}
          className="p-1.5 text-gray-400 hover:text-cyan-400 hover:bg-surface rounded transition"
          title="Reset zoom & pan"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* SVG Force-Directed Graph Viewport */}
      <div
        className="flex-1 w-full h-full relative cursor-grab active:cursor-grabbing overflow-hidden"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        {simNodes.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center text-gray-500 text-xs space-y-2 p-6">
            <Share2 className="w-8 h-8 text-gray-600 animate-pulse" />
            <p>
              {loading
                ? 'Loading LatticeDB knowledge graph...'
                : mode === 'traversed'
                ? 'No query path traversed yet. Execute a query in the Query Studio to see multi-hop graph hops.'
                : 'Knowledge graph is currently empty. Click "Ingest Document" above to populate entities and relations.'}
            </p>
          </div>
        ) : (
          <svg
            ref={svgRef}
            className="w-full h-full select-none"
            viewBox="0 0 600 500"
          >
            <defs>
              <marker
                id="arrowhead"
                viewBox="0 0 10 10"
                refX="22"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#4B5563" />
              </marker>
              <marker
                id="arrowhead-highlight"
                viewBox="0 0 10 10"
                refX="22"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#06B6D4" />
              </marker>
            </defs>

            <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
              {/* Edges */}
              {simEdges.map((edge, idx) => {
                const s = typeof edge.source === 'object' ? edge.source : null;
                const t = typeof edge.target === 'object' ? edge.target : null;
                if (!s || !t) return null;

                const isHighlighted =
                  connectedNodeIds?.has(edge.sourceId) && connectedNodeIds?.has(edge.targetId);
                const isDimmed = selectedNode && !isHighlighted;

                const midX = (s.x + t.x) / 2;
                const midY = (s.y + t.y) / 2;

                return (
                  <g key={idx} className={`transition-opacity duration-200 ${isDimmed ? 'opacity-15' : 'opacity-100'}`}>
                    <line
                      x1={s.x}
                      y1={s.y}
                      x2={t.x}
                      y2={t.y}
                      stroke={isHighlighted ? '#06B6D4' : '#374151'}
                      strokeWidth={isHighlighted ? 2 : 1}
                      markerEnd={isHighlighted ? 'url(#arrowhead-highlight)' : 'url(#arrowhead)'}
                    />
                    <text
                      x={midX}
                      y={midY - 4}
                      textAnchor="middle"
                      className="fill-gray-400 font-mono text-[9px] pointer-events-none select-none"
                    >
                      {edge.relationType}
                    </text>
                  </g>
                );
              })}

              {/* Nodes */}
              {simNodes.map((node) => {
                const isSelected = selectedNode?.id === node.id;
                const isConnected = connectedNodeIds?.has(node.id);
                const isDimmed = selectedNode && !isConnected;
                const theme = ENTITY_COLORS[node.label] || ENTITY_COLORS.Entity;

                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedNode(isSelected ? null : node);
                    }}
                    className={`cursor-pointer transition-all duration-200 ${
                      isDimmed ? 'opacity-20' : 'opacity-100'
                    }`}
                  >
                    <circle
                      r={isSelected ? 18 : 14}
                      fill={theme.fill}
                      stroke={isSelected ? '#FFFFFF' : theme.border}
                      strokeWidth={isSelected ? 2.5 : 1.5}
                      className="transition-all hover:scale-110"
                    />
                    <text
                      dy="24"
                      textAnchor="middle"
                      className="fill-gray-200 font-sans text-[10px] font-medium pointer-events-none select-none drop-shadow"
                    >
                      {node.name}
                    </text>
                    <text
                      dy="34"
                      textAnchor="middle"
                      className="fill-gray-500 font-mono text-[8px] pointer-events-none select-none uppercase tracking-wider"
                    >
                      {node.label}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
        )}
      </div>

      {/* Node Inspector Sliding Drawer */}
      {selectedNode && (
        <div className="absolute right-0 top-0 bottom-0 w-72 bg-card/95 border-l border-border p-4 shadow-2xl backdrop-blur flex flex-col justify-between z-20 animate-slideLeft">
          <div className="space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-border/80">
              <div className="flex items-center space-x-2">
                {getNodeIcon(selectedNode.label)}
                <span className="text-xs font-semibold text-gray-200 truncate max-w-[170px]">
                  {selectedNode.name}
                </span>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-gray-400 hover:text-gray-200 p-0.5 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center text-gray-400">
                <span>Node ID:</span>
                <span className="font-mono text-gray-200">{selectedNode.id}</span>
              </div>
              <div className="flex justify-between items-center text-gray-400">
                <span>Entity Label:</span>
                <span className="font-mono text-cyan-400 px-1.5 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/40 text-[10px]">
                  {selectedNode.label}
                </span>
              </div>
            </div>

            {/* Connected Relations List */}
            <div className="space-y-2">
              <h4 className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                Connected Relations ({selectedRelations.length})
              </h4>
              <div className="max-h-60 overflow-y-auto space-y-1.5">
                {selectedRelations.length === 0 ? (
                  <p className="text-[11px] text-gray-500">No relations attached.</p>
                ) : (
                  selectedRelations.map((rel, i) => {
                    const isOutgoing = rel.sourceId === selectedNode.id;
                    const otherNodeId = isOutgoing ? rel.targetId : rel.sourceId;
                    const otherNode = simNodes.find((n) => n.id === otherNodeId);
                    return (
                      <div
                        key={i}
                        className="bg-background/80 border border-border/80 rounded p-1.5 text-[11px] flex items-center justify-between font-mono"
                      >
                        <span className="text-gray-400 text-[10px]">
                          {isOutgoing ? '-> OUT' : '<- IN'}
                        </span>
                        <span className="text-cyan-300 font-medium">{rel.relationType}</span>
                        <span className="text-gray-300 truncate max-w-[80px]" title={otherNode?.name}>
                          {otherNode?.name || `id:${otherNodeId}`}
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-border/60 text-[10px] text-gray-500 flex items-center justify-between">
            <span>Embedded LatticeDB Node</span>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-cyan-400 hover:underline"
            >
              Clear selection
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
