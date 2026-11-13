// src/components/ProductCardTrio.tsx
import { Link } from 'react-router-dom';
import { LiquidGlassBadge } from './LiquidGlassBadge';

export function ProductCardTrio() {
  return (
    <section id="products" className="py-24 px-6 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="text-center max-w-2xl mx-auto mb-16">
        <LiquidGlassBadge variant="neutral" className="mb-4">
          Integrated Spatial Ecosystem
        </LiquidGlassBadge>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-glome-charcoal tracking-tight">
          Three Products, One Unified Reality
        </h2>
        <p className="text-glome-muted text-base mt-4 leading-relaxed">
          From consumer smartphone walkthroughs to BIM-grade construction analysis,
          explore our purpose-built spatial intelligence suite.
        </p>
      </div>

      {/* 3 Horizontal Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-7 items-stretch">
        
        {/* CARD 1: Glome Home Tour (Left — Live & Available) */}
        <div className="relative group rounded-3xl p-8 liquid-glass-card border-glome-orange/30 shadow-glass-glow-orange flex flex-col justify-between overflow-hidden bg-gradient-to-b from-white/90 via-white/80 to-glome-orange-tint/40">
          {/* Subtle Accent Glow Ring */}
          <div className="absolute -top-24 -right-24 w-48 h-48 bg-glome-orange/15 rounded-full blur-3xl pointer-events-none" />

          <div>
            {/* Header / Status */}
            <div className="flex items-center justify-between gap-3 mb-6">
              <LiquidGlassBadge variant="orange" pulse>
                Live & In Production
              </LiquidGlassBadge>
              <span className="text-xs font-mono text-glome-muted/80 tracking-wider">v1.2</span>
            </div>

            {/* Product Title & Brand */}
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-glome-orange to-glome-orange-dark flex items-center justify-center text-white shadow-md text-xl">
                🏠
              </div>
              <div>
                <h3 className="text-2xl font-bold text-glome-charcoal tracking-tight">
                  Glome Home Tour
                </h3>
                <p className="text-xs font-medium text-glome-orange">
                  Interactive 3D Virtual Walkthroughs
                </p>
              </div>
            </div>

            <p className="text-sm text-glome-muted leading-relaxed mb-6">
              Turn smartphone video walkthroughs into photorealistic, millimeter-accurate 3D digital twins.
              Zero LiDAR required, rendered in 6-DoF Gaussian Radiance and watertight metric meshes.
            </p>

            {/* Visual Preview Snippet */}
            <div className="p-4 rounded-2xl bg-white/70 border border-glome-stone/60 mb-6 backdrop-blur-sm space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-glome-charcoal font-semibold">Gaussian Splatting + Mesh</span>
                <span className="text-[11px] font-mono text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-100">
                  Sub-cm Precision
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] text-center font-medium text-glome-muted">
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/40">
                  <p className="font-bold text-glome-charcoal text-xs">6-DoF</p>
                  <p className="text-[10px] mt-0.5">Orbit & Walk</p>
                </div>
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/40">
                  <p className="font-bold text-glome-charcoal text-xs">2D Plans</p>
                  <p className="text-[10px] mt-0.5">Auto CAD/SVG</p>
                </div>
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/40">
                  <p className="font-bold text-glome-charcoal text-xs">WebGL</p>
                  <p className="text-[10px] mt-0.5">Zero Plugins</p>
                </div>
              </div>
            </div>

            {/* Feature Bullet List */}
            <ul className="space-y-2.5 text-xs text-glome-charcoal/90 mb-8">
              <li className="flex items-start gap-2.5">
                <span className="text-glome-orange font-bold text-sm leading-none mt-0.5">✓</span>
                <span><b>Smartphone capture</b>: Instant upload from Glome AR mobile app</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-orange font-bold text-sm leading-none mt-0.5">✓</span>
                <span><b>Dual 3D rendering</b>: Seamless toggle between Metric Mesh & Gaussian Splat</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-orange font-bold text-sm leading-none mt-0.5">✓</span>
                <span><b>Semantic room inventory</b>: Auto-classified furniture & spatial bounding boxes</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-orange font-bold text-sm leading-none mt-0.5">✓</span>
                <span><b>Automated 360 & flythrough</b>: 4K panoramas and cinematic video tours</span>
              </li>
            </ul>
          </div>

          {/* Action Link to the Home Tour platform */}
          <div className="pt-4 border-t border-glome-stone/40 flex items-center gap-3">
            <Link
              to="/dashboard"
              className="btn-primary w-full text-center justify-center !py-3 !text-sm group-hover:bg-glome-orange-dark shadow-md"
            >
              <span>Explore Home Tour Platform</span>
              <span className="transition-transform group-hover:translate-x-1">→</span>
            </Link>
          </div>
        </div>

        {/* CARD 2: Glome Home Design (Center — In Development) */}
        <div className="relative group rounded-3xl p-8 liquid-glass border-glome-stone/60 shadow-glass-sm flex flex-col justify-between overflow-hidden bg-gradient-to-b from-white/70 via-glome-cream/70 to-glome-moss-tint/30 opacity-90 hover:opacity-100 transition-opacity">
          {/* Status Ribbon Banner */}
          <div className="absolute top-6 right-6 z-20">
            <LiquidGlassBadge variant="moss">
              In Development
            </LiquidGlassBadge>
          </div>

          <div>
            {/* Header / Accent Bar */}
            <div className="flex items-center gap-2 mb-6">
              <span className="text-xs font-mono text-glome-muted uppercase tracking-wider">
                Generative Staging
              </span>
            </div>

            {/* Product Title & Brand */}
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-glome-moss to-glome-moss-dark flex items-center justify-center text-white shadow-md text-xl">
                🛋️
              </div>
              <div>
                <h3 className="text-2xl font-bold text-glome-charcoal tracking-tight">
                  Glome Home Design
                </h3>
                <p className="text-xs font-medium text-glome-moss">
                  Generative Spatial Staging & Renovation
                </p>
              </div>
            </div>

            <p className="text-sm text-glome-muted leading-relaxed mb-6">
              Virtually re-stage, furnish, and remodel empty or outdated interiors. Replace physical furniture
              with custom design catalogs while preserving exact spatial geometry and photorealistic lighting.
            </p>

            {/* In-Development Visual Representation */}
            <div className="p-4 rounded-2xl bg-white/60 border border-glome-stone/50 mb-6 backdrop-blur-sm space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-glome-charcoal font-semibold">Generative Interior AI</span>
                <span className="text-[11px] font-mono text-glome-moss bg-glome-moss/10 px-2 py-0.5 rounded-full border border-glome-moss/20">
                  Private Beta Q1 2027
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] text-center font-medium text-glome-muted">
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/30">
                  <p className="font-bold text-glome-charcoal text-xs">Declutter</p>
                  <p className="text-[10px] mt-0.5">Object removal</p>
                </div>
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/30">
                  <p className="font-bold text-glome-charcoal text-xs">AI Restyle</p>
                  <p className="text-[10px] mt-0.5">Scandinavian, Japandi</p>
                </div>
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/30">
                  <p className="font-bold text-glome-charcoal text-xs">Catalog</p>
                  <p className="text-[10px] mt-0.5">Real 3D assets</p>
                </div>
              </div>
            </div>

            {/* Feature Bullet List */}
            <ul className="space-y-2.5 text-xs text-glome-muted mb-8">
              <li className="flex items-start gap-2.5">
                <span className="text-glome-moss font-bold text-sm leading-none mt-0.5">✦</span>
                <span><b>Semantic inpainting</b>: Erase clutter and replace wall textures seamlessly</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-moss font-bold text-sm leading-none mt-0.5">✦</span>
                <span><b>BIM & retail catalog sync</b>: Place verified dimensional furnishings</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-moss font-bold text-sm leading-none mt-0.5">✦</span>
                <span><b>Physically based relighting</b>: Real-time sun angle & ambient occlusion</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-moss font-bold text-sm leading-none mt-0.5">✦</span>
                <span><b>Instant contractor specs</b>: Auto-calculate paint, flooring & tile square meterage</span>
              </li>
            </ul>
          </div>

          {/* Action Link (Disabled/Waitlist) */}
          <div className="pt-4 border-t border-glome-stone/40">
            <button
              onClick={() => alert('Glome Home Design is currently in private development. Contact us below to join the early preview waitlist!')}
              className="btn-moss w-full text-center justify-center !py-3 !text-sm shadow-sm"
            >
              <span>Join Design Waitlist</span>
              <span className="text-xs">🔒</span>
            </button>
          </div>
        </div>

        {/* CARD 3: Glome Build (Right — Coming Soon) */}
        <div className="relative group rounded-3xl p-8 liquid-glass border-glome-stone/60 shadow-glass-sm flex flex-col justify-between overflow-hidden bg-gradient-to-b from-white/70 via-glome-cream/70 to-glome-blueprint-tint/30 opacity-90 hover:opacity-100 transition-opacity">
          {/* Status Ribbon Banner */}
          <div className="absolute top-6 right-6 z-20">
            <LiquidGlassBadge variant="blueprint">
              Coming Soon
            </LiquidGlassBadge>
          </div>

          <div>
            {/* Header / Accent Bar */}
            <div className="flex items-center gap-2 mb-6">
              <span className="text-xs font-mono text-glome-muted uppercase tracking-wider">
                Construction Intelligence
              </span>
            </div>

            {/* Product Title & Brand */}
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-glome-blueprint to-glome-blueprint-dark flex items-center justify-center text-white shadow-md text-xl">
                📐
              </div>
              <div>
                <h3 className="text-2xl font-bold text-glome-charcoal tracking-tight">
                  Glome Build
                </h3>
                <p className="text-xs font-medium text-glome-blueprint">
                  As-Built Verification & BIM Auditing
                </p>
              </div>
            </div>

            <p className="text-sm text-glome-muted leading-relaxed mb-6">
              Automated structural auditing and volumetric tolerance inspection. Compare actual job-site
              scans directly against architectural CAD/BIM blueprints to detect construction deviations early.
            </p>

            {/* In-Development Visual Representation */}
            <div className="p-4 rounded-2xl bg-white/60 border border-glome-stone/50 mb-6 backdrop-blur-sm space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-glome-charcoal font-semibold">BIM Deviation Heatmaps</span>
                <span className="text-[11px] font-mono text-glome-blueprint bg-glome-blueprint/10 px-2 py-0.5 rounded-full border border-glome-blueprint/20">
                  Targeted Q2 2027
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] text-center font-medium text-glome-muted">
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/30">
                  <p className="font-bold text-glome-charcoal text-xs">±5mm</p>
                  <p className="text-[10px] mt-0.5">Tolerance map</p>
                </div>
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/30">
                  <p className="font-bold text-glome-charcoal text-xs">Revit/IFC</p>
                  <p className="text-[10px] mt-0.5">Direct overlay</p>
                </div>
                <div className="p-2 rounded-xl bg-glome-cream border border-glome-stone/30">
                  <p className="font-bold text-glome-charcoal text-xs">Slab Level</p>
                  <p className="text-[10px] mt-0.5">FF/FL reports</p>
                </div>
              </div>
            </div>

            {/* Feature Bullet List */}
            <ul className="space-y-2.5 text-xs text-glome-muted mb-8">
              <li className="flex items-start gap-2.5">
                <span className="text-glome-blueprint font-bold text-sm leading-none mt-0.5">✦</span>
                <span><b>Point-to-CAD verification</b>: Surface-to-model deviation heatmaps</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-blueprint font-bold text-sm leading-none mt-0.5">✦</span>
                <span><b>MEP conflict detection</b>: Check pipe, duct & conduit clearances in 3D</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-blueprint font-bold text-sm leading-none mt-0.5">✦</span>
                <span><b>Timeline chronometry</b>: Compare historical walkthroughs across build phases</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-glome-blueprint font-bold text-sm leading-none mt-0.5">✦</span>
                <span><b>Automated compliance logs</b>: Export signed PDF inspection certifications</span>
              </li>
            </ul>
          </div>

          {/* Action Link (Disabled/Enterprise) */}
          <div className="pt-4 border-t border-glome-stone/40">
            <button
              onClick={() => alert('Glome Build enterprise pilots are opening soon for construction contractors. Reach out via the contact form below.')}
              className="btn-blueprint w-full text-center justify-center !py-3 !text-sm shadow-sm"
            >
              <span>Request Architecture Pilot</span>
              <span className="text-xs">🔒</span>
            </button>
          </div>
        </div>

      </div>
    </section>
  );
}
