// src/components/FaqSection.tsx
import { useState } from 'react';
import { LiquidGlassBadge } from './LiquidGlassBadge';

export function FaqSection() {
  const [openIdx, setOpenIdx] = useState<number | null>(0);

  const faqs = [
    {
      q: 'What is the difference between Home Tour, Home Design, and Glome Build?',
      a: 'Glome Home Tour (available now) captures as-is properties and creates photorealistic 6-DoF walkthroughs and 2D floor plans. Glome Home Design (in development) focuses on generative interior renovation, virtual decluttering, and BIM furniture staging. Glome Build (coming soon) provides structural auditing, comparing job-site scans directly against architectural CAD/BIM blueprints to flag construction deviations.',
    },
    {
      q: 'How does Glome achieve metric scale accuracy without an expensive LiDAR rig?',
      a: 'Our pipeline leverages Depth Anything 3 (DA3) sliding-window depth ensembling, StableNormal surface orientation priors, and RANSAC structural alignment. During Stage 04 mesh elaboration, we snap floor-to-ceiling planes and resolve scale baselines against real-world camera trajectory baselines, delivering sub-centimeter dimensional fidelity from standard camera video.',
    },
    {
      q: 'What devices are supported for capturing property walkthroughs?',
      a: 'Any modern Android smartphone with ARCore or iPhone running the Glome Capture application. You do not need specialized LiDAR cameras, spherical tripods, or drone operators. A continuous steady walkthrough of the space is all that is required.',
    },
    {
      q: 'Do buyers or clients need to install any software or plugins to view 3D tours?',
      a: 'No. Glome tours run natively in modern web browsers (Chrome, Safari, Edge, Firefox) using standard WebGL and WebGPU. Users can orbit, walk through doors, inspect detected furniture, and switch between photorealistic Gaussian Splats and metric meshes on mobile phones, tablets, or desktops.',
    },
    {
      q: 'What 3D file formats can be exported from a finished reconstruction?',
      a: 'We support standard open formats: .OBJ and .PLY (with vertex colors and normals) for 3D meshes, .PLY for 3D Gaussian Splats, .SVG and .DXF for 2D architectural floor plans, and .MP4 for automated cinematic flythrough tours.',
    },
    {
      q: 'How long does the reconstruction pipeline take to process a typical apartment or house?',
      a: 'A typical multi-room residential walkthrough (300–400 keyframes) processes end-to-end in approximately 6 to 10 minutes on our accelerated AMD ROCm GPU infrastructure. Processing runs fully automatically in the background.',
    },
  ];

  return (
    <section id="faq" className="py-24 px-6 max-w-4xl mx-auto">
      {/* Section Header */}
      <div className="text-center mb-14">
        <LiquidGlassBadge variant="neutral" className="mb-4">
          Got Questions?
        </LiquidGlassBadge>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-glome-charcoal tracking-tight">
          Frequently Asked Questions
        </h2>
        <p className="text-glome-muted text-base mt-4 leading-relaxed">
          Everything you need to know about our 3D reconstruction pipeline, output formats, and ecosystem roadmap.
        </p>
      </div>

      {/* Accordion List */}
      <div className="space-y-4">
        {faqs.map((faq, idx) => {
          const isOpen = openIdx === idx;
          return (
            <div
              key={idx}
              className={`rounded-2xl transition-all duration-200 overflow-hidden ${
                isOpen
                  ? 'liquid-glass shadow-glass-sm border-glome-orange/30'
                  : 'bg-white/60 hover:bg-white/90 border border-glome-stone/50'
              }`}
            >
              <button
                onClick={() => setOpenIdx(isOpen ? null : idx)}
                className="w-full text-left p-6 flex items-center justify-between gap-4 font-bold text-glome-charcoal text-base"
              >
                <span>{faq.q}</span>
                <span
                  className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 text-sm font-semibold transition-transform duration-200 ${
                    isOpen
                      ? 'bg-glome-orange text-white rotate-45'
                      : 'bg-glome-cream text-glome-muted'
                  }`}
                >
                  +
                </span>
              </button>

              {isOpen && (
                <div className="px-6 pb-6 pt-1 text-sm text-glome-muted leading-relaxed border-t border-glome-stone/30">
                  {faq.a}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
