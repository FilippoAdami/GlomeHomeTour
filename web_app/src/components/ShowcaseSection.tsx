// src/components/ShowcaseSection.tsx
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { LiquidGlassBadge } from './LiquidGlassBadge';

export function ShowcaseSection() {
  const [activeTab, setActiveTab] = useState<'splat' | 'mesh' | 'floorplan' | 'segmentation'>('splat');

  const tabs = [
    { id: 'splat', label: '✨ Gaussian Radiance', subtitle: 'Photoreal 6-DoF' },
    { id: 'mesh', label: '📐 Metric TSDF Mesh', subtitle: 'Watertight Geometry' },
    { id: 'floorplan', label: '🗺️ 2D Floor Plans', subtitle: 'Auto CAD & Dimensions' },
    { id: 'segmentation', label: '🏷️ 3D Segmentation', subtitle: 'Panoptic Objects' },
  ] as const;

  return (
    <section id="showcase" className="py-24 px-6 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="text-center max-w-3xl mx-auto mb-16">
        <LiquidGlassBadge variant="orange" className="mb-4">
          Proof of Performance
        </LiquidGlassBadge>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-glome-charcoal tracking-tight">
          What We Build: From Phone Video to Spatial Twin
        </h2>
        <p className="text-glome-muted text-base mt-4 leading-relaxed">
          Examine the actual outputs produced by our backend pipeline from a single 90-second smartphone capture.
        </p>
      </div>

      {/* Interactive Showcase Box */}
      <div className="liquid-glass rounded-3xl p-6 sm:p-10 border-glome-stone/60 shadow-glass-lg overflow-hidden bg-gradient-to-b from-white/90 to-glome-cream/70">
        
        {/* Navigation Tabs */}
        <div className="flex flex-wrap items-center justify-center gap-2 pb-8 border-b border-glome-stone/40">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-5 py-3 rounded-2xl text-xs font-semibold transition-all flex flex-col items-center sm:items-start ${
                  isActive
                    ? 'bg-glome-orange text-white shadow-md scale-[1.02]'
                    : 'bg-white/60 text-glome-muted hover:text-glome-charcoal hover:bg-white border border-glome-stone/40'
                }`}
              >
                <span>{tab.label}</span>
                <span className={`text-[10px] mt-0.5 font-normal ${isActive ? 'text-white/80' : 'text-glome-muted/70'}`}>
                  {tab.subtitle}
                </span>
              </button>
            );
          })}
        </div>

        {/* Tab Content Display */}
        <div className="py-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          
          {/* Left: Spec details */}
          <div className="lg:col-span-5 space-y-5">
            <span className="text-xs font-mono uppercase tracking-wider text-glome-orange font-bold">
              Case Study: Master Bedroom Suite
            </span>
            <h3 className="text-2xl font-bold text-glome-charcoal tracking-tight">
              {activeTab === 'splat' && 'Neural Radiance Fields via FastGS'}
              {activeTab === 'mesh' && 'Watertight Closed TSDF Mesh'}
              {activeTab === 'floorplan' && 'Sub-Centimeter Floor Plan Projections'}
              {activeTab === 'segmentation' && 'DINOv2 + SAM2 3D Semantic Clusters'}
            </h3>
            
            <p className="text-sm text-glome-muted leading-relaxed">
              {activeTab === 'splat' &&
                'Trained in under 150 seconds on AMD ROCm GPUs with DN-Splatter depth and surface normal supervision. Delivers photorealistic reflections on polished hardwood, mirror specularities, and intricate textile draping.'}
              {activeTab === 'mesh' &&
                'Open3D scalable TSDF fusion followed by our custom floater-removal and planar boundary hole sealing. Decimated to 150k polygons for instantaneous 60 FPS WebGL rendering without losing wall squareness.'}
              {activeTab === 'floorplan' &&
                'Top-down 5mm raster projection fitted with constrained orthogonal boundaries. Automatically detects and measures doors, windows, ceiling clearances, and net usable area.'}
              {activeTab === 'segmentation' &&
                '2D image segment proposals back-projected into 3D point clusters. Classified with EmbeddingGemma into 21 structural and furniture categories with exact bounding volumes.'}
            </p>

            <div className="grid grid-cols-2 gap-3 pt-2">
              <div className="p-3.5 rounded-xl bg-white/70 border border-glome-stone/40">
                <p className="text-[11px] text-glome-muted">Acquisition Time</p>
                <p className="text-base font-bold text-glome-charcoal mt-0.5">92 Seconds</p>
              </div>
              <div className="p-3.5 rounded-xl bg-white/70 border border-glome-stone/40">
                <p className="text-[11px] text-glome-muted">Registered Frames</p>
                <p className="text-base font-bold text-glome-charcoal mt-0.5">346 COLMAP Poses</p>
              </div>
              <div className="p-3.5 rounded-xl bg-white/70 border border-glome-stone/40">
                <p className="text-[11px] text-glome-muted">Total Processing</p>
                <p className="text-base font-bold text-glome-charcoal mt-0.5">7.4 Minutes</p>
              </div>
              <div className="p-3.5 rounded-xl bg-white/70 border border-glome-stone/40">
                <p className="text-[11px] text-glome-muted">Output Formats</p>
                <p className="text-base font-bold text-glome-charcoal mt-0.5">PLY, OBJ, SVG, GLB</p>
              </div>
            </div>

            <div className="pt-3">
              <Link
                to="/viewer/demo-bedroom"
                className="btn-primary text-xs !py-3 !px-6 shadow-md"
              >
                <span>Inspect Live in 3D Viewer</span>
                <span>🌐</span>
              </Link>
            </div>
          </div>

          {/* Right: Visual Interactive Card Representation */}
          <div className="lg:col-span-7">
            <div className="relative rounded-2xl overflow-hidden border border-glome-stone/60 shadow-glass-md bg-slate-950 aspect-[16/10] flex items-center justify-center p-6 text-white group">
              
              {/* Background preview based on active tab */}
              {activeTab === 'splat' && (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 rounded-2xl bg-glome-orange/20 border border-glome-orange/40 flex items-center justify-center text-3xl mx-auto shadow-[0_0_30px_rgba(217,93,40,0.4)]">
                    ✨
                  </div>
                  <div>
                    <h4 className="text-lg font-bold">Gaussian Radiance Field (Demo Scene)</h4>
                    <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                      Real-time view-dependent lighting, soft shadows, and sub-pixel edge sharpness.
                    </p>
                  </div>
                  <div className="flex items-center justify-center gap-2 pt-2">
                    <span className="px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-[11px] text-slate-300">
                      WebGL Splat Renderer
                    </span>
                    <span className="px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-500/30 text-[11px] text-emerald-300">
                      60 FPS Active
                    </span>
                  </div>
                </div>
              )}

              {activeTab === 'mesh' && (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 rounded-2xl bg-blue-500/20 border border-blue-500/40 flex items-center justify-center text-3xl mx-auto shadow-[0_0_30px_rgba(59,130,246,0.4)]">
                    📐
                  </div>
                  <div>
                    <h4 className="text-lg font-bold">Closed TSDF Triangle Mesh</h4>
                    <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                      Watertight surfaces with coplanar color refinement and synthetic hole infill.
                    </p>
                  </div>
                  <div className="flex items-center justify-center gap-2 pt-2">
                    <span className="px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-[11px] text-slate-300">
                      150,000 Polygons
                    </span>
                    <span className="px-3 py-1 rounded-full bg-blue-950/80 border border-blue-500/30 text-[11px] text-blue-300">
                      PBR Shaded
                    </span>
                  </div>
                </div>
              )}

              {activeTab === 'floorplan' && (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 rounded-2xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-3xl mx-auto shadow-[0_0_30px_rgba(245,158,11,0.4)]">
                    🗺️
                  </div>
                  <div>
                    <h4 className="text-lg font-bold">2D Metric Orthogonal Plan</h4>
                    <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                      Vectorized room boundaries, door openings, and ceiling clearance elevation stamps.
                    </p>
                  </div>
                  <div className="flex items-center justify-center gap-2 pt-2">
                    <span className="px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-[11px] text-slate-300">
                      5mm Raster Fit
                    </span>
                    <span className="px-3 py-1 rounded-full bg-amber-950/80 border border-amber-500/30 text-[11px] text-amber-300">
                      CAD Exportable
                    </span>
                  </div>
                </div>
              )}

              {activeTab === 'segmentation' && (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 rounded-2xl bg-purple-500/20 border border-purple-500/40 flex items-center justify-center text-3xl mx-auto shadow-[0_0_30px_rgba(168,85,247,0.4)]">
                    🏷️
                  </div>
                  <div>
                    <h4 className="text-lg font-bold">Semantic 3D Spatial Inventory</h4>
                    <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                      Automatic identification of furniture, doors, lighting fixtures, and dimensions.
                    </p>
                  </div>
                  <div className="flex items-center justify-center gap-2 pt-2">
                    <span className="px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-[11px] text-slate-300">
                      21 Categories
                    </span>
                    <span className="px-3 py-1 rounded-full bg-purple-950/80 border border-purple-500/30 text-[11px] text-purple-300">
                      EmbeddingGemma
                    </span>
                  </div>
                </div>
              )}

              {/* Bottom overlay badge */}
              <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between text-[11px] text-slate-400 bg-slate-900/80 backdrop-blur-md px-3.5 py-2 rounded-xl border border-slate-800">
                <span>Model ID: <code className="text-slate-300">demo-bedroom</code></span>
                <span className="text-glome-orange font-semibold">Live in Local Database</span>
              </div>
            </div>
          </div>

        </div>

      </div>
    </section>
  );
}
