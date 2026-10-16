// src/pages/LandingPage.tsx

import { Link } from 'react-router-dom'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white">
      {/* Topbar */}
      <header className="border-b border-gray-100">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <span className="font-bold text-glome-navy text-xl tracking-tight">Glome</span>
          <div className="flex items-center gap-4">
            <Link to="/sign-in" className="text-sm text-gray-500 hover:text-glome-navy transition-colors">
              Sign in
            </Link>
            <Link to="/sign-up" className="btn-primary text-sm">
              Get started
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-20 text-center">
        <span className="inline-block bg-glome-blue/10 text-glome-blue text-xs font-semibold px-3 py-1 rounded-full mb-6">
          3D Home Tours, Reinvented
        </span>
        <h1 className="text-5xl font-extrabold text-glome-navy leading-tight mb-6">
          Capture any property.<br />
          <span className="text-glome-blue">Share it in 3D.</span>
        </h1>
        <p className="text-gray-500 text-lg max-w-2xl mx-auto mb-10">
          Glome turns a standard smartphone walkthrough into a photorealistic,
          interactive 3D property tour — no LiDAR, no specialist hardware required.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Link to="/sign-up" className="btn-primary px-8 py-3 text-base">
            Start for free
          </Link>
          <a href="#how-it-works" className="btn-secondary px-8 py-3 text-base">
            See how it works
          </a>
        </div>
      </section>

      {/* Feature strip */}
      <section id="how-it-works" className="bg-glome-off-white py-20">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-glome-navy text-center mb-12">
            From walkthrough to 3D twin in minutes
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Capture',
                desc: 'Walk through the property with the Glome mobile app. Guided overlays ensure every corner is covered.',
              },
              {
                step: '02',
                title: 'Process',
                desc: 'Upload the archive. Our backend runs Gaussian Splatting, depth estimation, and floor plan extraction automatically.',
              },
              {
                step: '03',
                title: 'Share',
                desc: 'Send a link. Buyers explore the photorealistic 3D tour and floor plan from any browser, no plugin required.',
              },
            ].map(({ step, title, desc }) => (
              <div key={step} className="card p-8">
                <span className="text-4xl font-extrabold text-glome-blue/20">{step}</span>
                <h3 className="text-lg font-bold text-glome-navy mt-2 mb-2">{title}</h3>
                <p className="text-gray-500 text-sm leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 bg-glome-navy text-center">
        <h2 className="text-3xl font-bold text-white mb-4">
          Ready to list in 3D?
        </h2>
        <p className="text-white/60 mb-8">
          Free to start. No credit card required.
        </p>
        <Link to="/sign-up" className="btn-primary bg-glome-gold text-glome-navy hover:bg-yellow-400 px-10 py-3 text-base">
          Create your account
        </Link>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-100 py-8">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between text-sm text-gray-400">
          <span>© 2026 Glome. All rights reserved.</span>
          <span>Built with ❤️ for real estate professionals</span>
        </div>
      </footer>
    </div>
  )
}
