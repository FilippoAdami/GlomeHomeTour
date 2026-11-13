// src/components/CompanyFacts.tsx
import { LiquidGlassBadge } from './LiquidGlassBadge';

export function CompanyFacts() {
  const stats = [
    {
      value: '< 8 mm',
      label: 'Metric Spatial Accuracy',
      desc: 'Rigid scale calibration and closed TSDF surfaces without expensive LiDAR rigs.',
    },
    {
      value: '100%',
      label: 'Standard Phone Cameras',
      desc: 'Image-only and ARCore tracking turns everyday smartphones into 3D survey tools.',
    },
    {
      value: '10x Faster',
      label: 'GPU Pipeline Speedup',
      desc: 'Accelerated PyTorch + ROCm neural radiance training with DN-Splatter supervision.',
    },
    {
      value: '6-DoF',
      label: 'Unconstrained Freedom',
      desc: 'Free roaming orbit and walkthrough navigation unlike traditional pinned 360 bubbles.',
    },
  ];

  const clientTypes = [
    {
      role: 'High-End Real Estate',
      icon: '🏢',
      description: 'Ultra-luxury residential brokerages needing photorealistic walkthroughs for international buyers.',
    },
    {
      role: 'Architectural Studios',
      icon: '🏛️',
      description: 'Preservationists and designers documenting existing site geometry with exact 2D floor plans.',
    },
    {
      role: 'Construction Contractors',
      icon: '🏗️',
      description: 'Project managers conducting remote progress audits, subcontractor checkoffs, and MEP verification.',
    },
    {
      role: 'Hospitality & Venues',
      icon: '✨',
      description: 'Boutique hotels and event spaces creating interactive digital twins for direct web bookings.',
    },
  ];

  return (
    <section id="capabilities" className="py-24 px-6 max-w-7xl mx-auto">
      {/* Metrics Section */}
      <div className="text-center max-w-3xl mx-auto mb-16">
        <LiquidGlassBadge variant="neutral" className="mb-4">
          Architectural Engineering Standards
        </LiquidGlassBadge>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-glome-charcoal tracking-tight">
          Engineering Rigor in Every Dimension
        </h2>
        <p className="text-glome-muted text-base mt-4 leading-relaxed">
          Glome bridges the gap between game-engine photorealism and surveyor-grade dimensional accuracy.
          Here is how our technology stands apart from legacy photogrammetry.
        </p>
      </div>

      {/* Grid of Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-20">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="liquid-glass-card p-6 rounded-2xl flex flex-col justify-between border-glome-stone/60"
          >
            <div>
              <p className="text-3xl sm:text-4xl font-extrabold text-glome-orange tracking-tight">
                {stat.value}
              </p>
              <h3 className="text-sm font-bold text-glome-charcoal mt-2">
                {stat.label}
              </h3>
            </div>
            <p className="text-xs text-glome-muted mt-3 leading-relaxed">
              {stat.desc}
            </p>
          </div>
        ))}
      </div>

      {/* Who We Work With */}
      <div className="p-8 sm:p-12 rounded-3xl liquid-glass border-glome-stone/60 shadow-glass-md bg-gradient-to-b from-white/80 to-glome-cream/60">
        <div className="max-w-2xl mb-10">
          <span className="text-xs font-mono uppercase tracking-wider text-glome-orange font-semibold">
            Industry Solutions
          </span>
          <h3 className="text-2xl sm:text-3xl font-bold text-glome-charcoal mt-1 tracking-tight">
            Who We Partner With
          </h3>
          <p className="text-sm text-glome-muted mt-2">
            Glome deploys custom 3D capture and reconstruction solutions tailored for premier industry leaders.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {clientTypes.map((client) => (
            <div
              key={client.role}
              className="p-5 rounded-2xl bg-white/70 border border-glome-stone/50 backdrop-blur-sm flex flex-col justify-between"
            >
              <div>
                <span className="text-2xl">{client.icon}</span>
                <h4 className="font-bold text-glome-charcoal text-sm mt-3">
                  {client.role}
                </h4>
                <p className="text-xs text-glome-muted mt-2 leading-relaxed">
                  {client.description}
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-glome-stone/30 flex items-center text-[11px] font-semibold text-glome-orange">
                <span>Enterprise ready</span>
                <span className="ml-1">✓</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
