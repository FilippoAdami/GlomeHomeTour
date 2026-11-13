// src/components/AgencyFooter.tsx
import React, { useState } from 'react';
import { Link } from 'react-router-dom';

export function AgencyFooter() {
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);

  function handleSubscribe(e: React.FormEvent) {
    e.preventDefault();
    if (!email) return;
    setSubmitted(true);
    setTimeout(() => {
      setEmail('');
      setSubmitted(false);
    }, 4000);
  }

  return (
    <footer className="mt-20 border-t border-glome-stone/50 bg-gradient-to-b from-glome-cream via-white to-glome-cream/80 pt-16 pb-12 px-6">
      <div className="max-w-7xl mx-auto space-y-16">
        
        {/* Top Tier: Brand + Inquiry Box */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
          
          {/* Brand info */}
          <div className="lg:col-span-5 space-y-4">
            <Link to="/" className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-glome-orange to-glome-orange-light flex items-center justify-center text-white shadow-sm font-bold">
                G
              </div>
              <span className="font-extrabold text-glome-charcoal text-xl tracking-tight">
                Glome
              </span>
            </Link>
            <p className="text-sm text-glome-muted leading-relaxed max-w-sm">
              The premier 3D reconstruction and spatial intelligence agency.
              Engineering photorealistic digital twins, CAD-grade metric meshes,
              and generative architectural environments.
            </p>
            <div className="flex items-center gap-3 text-xs text-glome-muted pt-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200/60 font-medium">
                ● All Systems Operational
              </span>
              <span>GPU Cluster ROCm 7.1</span>
            </div>
          </div>

          {/* Contact / Newsletter Box */}
          <div className="lg:col-span-7 p-6 sm:p-8 rounded-3xl liquid-glass border-glome-stone/60 shadow-glass-sm">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
              <div>
                <h4 className="text-base font-bold text-glome-charcoal tracking-tight">
                  Stay Informed on Glome Design & Build
                </h4>
                <p className="text-xs text-glome-muted mt-0.5">
                  Receive private beta invitations, technical benchmarks, and architecture case studies.
                </p>
              </div>
            </div>

            <form onSubmit={handleSubscribe} className="flex flex-col sm:flex-row gap-2.5">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="architect@firm.com"
                required
                className="input flex-1 !text-xs !py-3"
              />
              <button type="submit" className="btn-primary !text-xs !py-3 !px-6 shrink-0">
                {submitted ? '✓ Subscribed!' : 'Request Updates'}
              </button>
            </form>
          </div>

        </div>

        {/* Middle Tier: Column Links */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-8 pt-6 border-t border-glome-stone/40 text-xs">
          
          {/* Products */}
          <div className="space-y-3">
            <h5 className="font-bold text-glome-charcoal uppercase tracking-wider text-[11px]">
              Products
            </h5>
            <ul className="space-y-2 text-glome-muted">
              <li>
                <Link to="/dashboard" className="hover:text-glome-orange transition-colors flex items-center gap-1.5">
                  <span>Glome Home Tour</span>
                  <span className="text-[10px] font-bold text-glome-orange px-1.5 py-0.2 bg-glome-orange/10 rounded">LIVE</span>
                </Link>
              </li>
              <li>
                <span className="text-glome-muted/70 flex items-center gap-1.5 cursor-not-allowed">
                  <span>Glome Home Design</span>
                  <span className="text-[10px] text-glome-moss font-semibold">DEV</span>
                </span>
              </li>
              <li>
                <span className="text-glome-muted/70 flex items-center gap-1.5 cursor-not-allowed">
                  <span>Glome Build</span>
                  <span className="text-[10px] text-glome-blueprint font-semibold">SOON</span>
                </span>
              </li>
              <li>
                <Link to="/viewer/demo-bedroom" className="hover:text-glome-orange transition-colors">
                  Interactive 3D Demo
                </Link>
              </li>
            </ul>
          </div>

          {/* Technology */}
          <div className="space-y-3">
            <h5 className="font-bold text-glome-charcoal uppercase tracking-wider text-[11px]">
              Technology
            </h5>
            <ul className="space-y-2 text-glome-muted">
              <li><span>3D Gaussian Splatting</span></li>
              <li><span>Scalable TSDF Mesh Fusion</span></li>
              <li><span>Depth Anything 3 (DA3)</span></li>
              <li><span>DINOv2 Panoptic Clusters</span></li>
              <li><span>Orthogonal Floor Plan Vectorization</span></li>
            </ul>
          </div>

          {/* Agency & Company */}
          <div className="space-y-3">
            <h5 className="font-bold text-glome-charcoal uppercase tracking-wider text-[11px]">
              Agency
            </h5>
            <ul className="space-y-2 text-glome-muted">
              <li><a href="#mission" className="hover:text-glome-orange transition-colors">Company Mission</a></li>
              <li><a href="#capabilities" className="hover:text-glome-orange transition-colors">Enterprise Capabilities</a></li>
              <li><a href="#showcase" className="hover:text-glome-orange transition-colors">Client Showcase</a></li>
              <li><a href="#faq" className="hover:text-glome-orange transition-colors">FAQ</a></li>
            </ul>
          </div>

          {/* Direct Contacts */}
          <div className="space-y-3">
            <h5 className="font-bold text-glome-charcoal uppercase tracking-wider text-[11px]">
              Contacts & Info
            </h5>
            <ul className="space-y-2 text-glome-muted">
              <li>
                <span className="font-medium text-glome-charcoal block">Inquiries:</span>
                <a href="mailto:inquiries@glome3d.com" className="hover:text-glome-orange transition-colors">
                  inquiries@glome3d.com
                </a>
              </li>
              <li>
                <span className="font-medium text-glome-charcoal block">Engineering:</span>
                <a href="mailto:engineering@glome3d.com" className="hover:text-glome-orange transition-colors">
                  engineering@glome3d.com
                </a>
              </li>
              <li>
                <span className="font-medium text-glome-charcoal block">Studio:</span>
                <span>London & Milan</span>
              </li>
            </ul>
          </div>

        </div>

        {/* Bottom Tier: Copyright & Disclaimers */}
        <div className="pt-8 border-t border-glome-stone/30 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-glome-muted">
          <div>
            © 2026 Glome 3D Reconstruction Agency. All rights reserved.
          </div>
          <div className="flex items-center gap-6">
            <span className="hover:text-glome-charcoal cursor-pointer">Privacy Policy</span>
            <span className="hover:text-glome-charcoal cursor-pointer">Terms of Service</span>
            <span className="hover:text-glome-charcoal cursor-pointer">Security Audits</span>
          </div>
        </div>

      </div>
    </footer>
  );
}
