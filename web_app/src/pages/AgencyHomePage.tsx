// src/pages/AgencyHomePage.tsx
import { Link } from 'react-router-dom';
import { NavigationBar } from '../components/NavigationBar';
import { ProductCardTrio } from '../components/ProductCardTrio';
import { CompanyFacts } from '../components/CompanyFacts';
import { ShowcaseSection } from '../components/ShowcaseSection';
import { FaqSection } from '../components/FaqSection';
import { AgencyFooter } from '../components/AgencyFooter';
import { LiquidGlassBadge } from '../components/LiquidGlassBadge';

export default function AgencyHomePage() {
  return (
    <div className="min-h-screen bg-glome-cream text-glome-charcoal selection:bg-glome-orange/20 selection:text-glome-orange-dark relative overflow-hidden">
      {/* Ambient background refractive gradient blurs */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[550px] bg-gradient-to-b from-glome-orange/10 via-glome-cream to-transparent rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-[800px] left-0 w-[500px] h-[500px] bg-glome-moss/5 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-[1400px] right-0 w-[600px] h-[600px] bg-glome-blueprint/5 rounded-full blur-3xl pointer-events-none -z-10" />

      {/* Floating Liquid Glass Navigation Bar */}
      <NavigationBar />

      {/* Hero & Agency Mission Section */}
      <section id="mission" className="pt-36 sm:pt-44 pb-20 px-6 max-w-6xl mx-auto text-center">
        <div className="inline-flex items-center gap-2 mb-6">
          <LiquidGlassBadge variant="orange" pulse>
            Next-Gen 3D Reconstruction Agency
          </LiquidGlassBadge>
        </div>

        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold text-glome-charcoal tracking-tight leading-[1.08] max-w-4xl mx-auto">
          Spatial Reality, <br />
          <span className="bg-gradient-to-r from-glome-orange via-glome-orange-light to-glome-orange-dark bg-clip-text text-transparent">
            Perfected in 3D.
          </span>
        </h1>

        <p className="text-glome-muted text-lg sm:text-xl max-w-2xl mx-auto mt-6 leading-relaxed font-normal">
          Glome turns standard smartphone walkthroughs into surveyor-grade metric meshes,
          photorealistic Gaussian radiance fields, and spatial intelligence models.
          Zero LiDAR rigs. Zero specialist hardware.
        </p>

        {/* Primary CTA Cluster */}
        <div className="flex flex-wrap items-center justify-center gap-4 mt-10">
          <Link
            to="/dashboard"
            className="btn-primary !px-8 !py-3.5 !text-sm shadow-md hover:shadow-lg"
          >
            <span>Launch Glome Home Tour</span>
            <span className="text-base">→</span>
          </Link>
          <a
            href="#products"
            className="btn-secondary !px-8 !py-3.5 !text-sm"
          >
            Explore Agency Products
          </a>
        </div>

        {/* Interactive Hero Visual Glass Card */}
        <div className="mt-16 p-4 sm:p-6 rounded-3xl liquid-glass border-glome-stone/60 shadow-glass-md max-w-4xl mx-auto bg-gradient-to-b from-white/90 via-white/80 to-glome-cream/70 text-left">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-glome-stone/40">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-glome-orange-tint border border-glome-orange/30 flex items-center justify-center text-glome-orange font-bold text-lg">
                📐
              </div>
              <div>
                <h3 className="text-sm font-bold text-glome-charcoal">
                  Multi-Modal Reconstruction Engine
                </h3>
                <p className="text-xs text-glome-muted">
                  COLMAP 6-DoF • StableNormal • Depth Anything 3 • FastGS Radiance
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-emerald-700 bg-emerald-50 border border-emerald-200/60 px-2.5 py-1 rounded-full font-semibold">
                ● 100% Watertight Mesh Verified
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 text-xs">
            <div className="p-3.5 rounded-xl bg-white/70 border border-glome-stone/40">
              <span className="text-glome-orange font-bold text-sm block">Glome Home Tour</span>
              <p className="text-glome-muted text-[11px] mt-1">
                6-DoF Gaussian virtual tours & automated CAD floor plans. Available today.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-white/70 border border-glome-stone/40">
              <span className="text-glome-moss font-bold text-sm block">Glome Home Design</span>
              <p className="text-glome-muted text-[11px] mt-1">
                Generative AI staging, semantic decluttering & BIM material swaps.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-white/70 border border-glome-stone/40">
              <span className="text-glome-blueprint font-bold text-sm block">Glome Build</span>
              <p className="text-glome-muted text-[11px] mt-1">
                Point-to-CAD BIM deviation auditing & construction tolerance inspection.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* The 3 Products Horizontal Trio Section */}
      <ProductCardTrio />

      {/* Company Facts & Capabilities */}
      <CompanyFacts />

      {/* Real Deliverables & Work Showcase */}
      <ShowcaseSection />

      {/* FAQ Accordion */}
      <FaqSection />

      {/* Comprehensive Footer */}
      <AgencyFooter />
    </div>
  );
}
