// src/pages/LandingPage.tsx

import { Link } from 'react-router-dom'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-glome-cream text-glome-charcoal selection:bg-glome-orange/20 selection:text-glome-orange-dark">
      {/* Topbar */}
      <header className="border-b border-glome-stone/40 bg-white/70 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/" className="text-xs text-glome-muted hover:text-glome-charcoal transition-colors flex items-center gap-1">
              <span>← Glome Agency</span>
            </Link>
            <span className="text-glome-stone">|</span>
            <span className="font-bold text-glome-charcoal text-lg tracking-tight flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-glome-orange inline-block" />
              Glome Home Tour
            </span>
          </div>

          <div className="flex items-center gap-4">
            <Link to="/sign-in" className="text-xs font-semibold text-glome-muted hover:text-glome-charcoal transition-colors">
              Sign In
            </Link>
            <Link to="/sign-up" className="btn-primary !py-2 !px-4 text-xs">
              Get Started Free
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-20 pb-20 text-center">
        <span className="inline-block bg-glome-orange/10 text-glome-orange border border-glome-orange/20 text-xs font-semibold px-3.5 py-1 rounded-full mb-6 shadow-sm">
          3D Virtual Walkthroughs & Automated Floor Plans
        </span>
        <h1 className="text-5xl sm:text-6xl font-extrabold text-glome-charcoal leading-tight mb-6 tracking-tight">
          Capture any property.<br />
          <span className="text-glome-orange">Share it in photoreal 3D.</span>
        </h1>
        <p className="text-glome-muted text-lg max-w-2xl mx-auto mb-10 leading-relaxed">
          Glome turns a standard smartphone walkthrough into a photorealistic,
          interactive 3D property tour — no LiDAR, no specialist hardware required.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Link to="/sign-up" className="btn-primary px-8 py-3.5 text-sm shadow-md">
            Start for free
          </Link>
          <Link to="/viewer/demo-bedroom" className="btn-secondary px-8 py-3.5 text-sm">
            View Live 3D Demo 🌐
          </Link>
        </div>
      </section>

      {/* Feature strip */}
      <section id="how-it-works" className="bg-glome-cream-dark/50 border-y border-glome-stone/40 py-20">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-glome-charcoal text-center mb-12 tracking-tight">
            From Walkthrough to 3D Twin in Minutes
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Capture',
                desc: 'Walk through the property with the Glome mobile app. Guided AR overlays ensure every room angle is registered.',
              },
              {
                step: '02',
                title: 'Process',
                desc: 'Upload the archive. Our GPU pipeline runs Gaussian Splatting, metric depth estimation, and TSDF mesh hole-filling.',
              },
              {
                step: '03',
                title: 'Share',
                desc: 'Send a link. Buyers explore the photorealistic 3D tour, semantic room inventory, and floor plan from any web browser.',
              },
            ].map(({ step, title, desc }) => (
              <div key={step} className="liquid-glass-card p-8 rounded-3xl border-glome-stone/60">
                <span className="text-4xl font-extrabold text-glome-orange/30 font-mono">{step}</span>
                <h3 className="text-lg font-bold text-glome-charcoal mt-2 mb-2">{title}</h3>
                <p className="text-glome-muted text-sm leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 bg-gradient-to-br from-glome-charcoal via-glome-charcoal-light to-glome-charcoal text-center px-6 text-white">
        <h2 className="text-3xl font-bold mb-4 tracking-tight">
          Ready to List in 3D?
        </h2>
        <p className="text-white/70 max-w-md mx-auto mb-8 text-sm leading-relaxed">
          Free to start in local mode. Generate your first interactive digital twin today.
        </p>
        <Link to="/sign-up" className="btn-primary !bg-glome-orange hover:!bg-glome-orange-dark px-10 py-3.5 text-sm shadow-lg">
          Create Free Account
        </Link>
      </section>

      {/* Footer */}
      <footer className="border-t border-glome-stone/40 py-8 bg-glome-cream">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between text-xs text-glome-muted gap-4">
          <Link to="/" className="hover:text-glome-charcoal transition-colors">
            ← Return to Glome Agency Master Page
          </Link>
          <span>© 2026 Glome Technologies Inc. All rights reserved.</span>
        </div>
      </footer>
    </div>
  );
}
