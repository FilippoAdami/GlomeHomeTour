// src/pages/DashboardPage.tsx
import { useState } from 'react';
import { useOrg } from '../hooks/useOrg';
import { useProperties } from '../hooks/useProperties';
import { PropertyCard } from '../components/PropertyCard';
import { UploadModal } from '../components/UploadModal';

export default function DashboardPage() {
  const { org, loading: orgLoading } = useOrg();
  const { properties, loading: propsLoading, refresh } = useProperties(org?.id ?? null);
  const [showUpload, setShowUpload] = useState(false);

  const loading = orgLoading || propsLoading;

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex items-center justify-between mb-8 pb-4 border-b border-glome-stone/40">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2.5 h-2.5 rounded-full bg-glome-orange inline-block" />
            <h1 className="text-2xl font-bold text-glome-charcoal tracking-tight">Properties</h1>
          </div>
          <p className="text-glome-muted text-xs">
            {properties.length} {properties.length === 1 ? 'property' : 'properties'} in your organization
          </p>
        </div>
        <button
          onClick={() => setShowUpload(true)}
          className="btn-primary shadow-sm"
        >
          + New Scan Archive
        </button>
      </div>

      {/* Content */}
      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="card p-6 h-52 animate-pulse bg-white/50 border-glome-stone/30" />
          ))}
        </div>
      ) : properties.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center liquid-glass rounded-3xl p-8 border-glome-stone/60">
          <div className="text-5xl mb-4">🏠</div>
          <h2 className="text-lg font-bold text-glome-charcoal mb-2">No properties yet</h2>
          <p className="text-glome-muted text-xs max-w-sm leading-relaxed">
            Upload a scan archive from the Glome mobile app to generate your photorealistic 3D property tour.
          </p>
          <button onClick={() => setShowUpload(true)} className="btn-primary mt-6 text-xs !py-3 !px-6">
            Upload Your First Scan
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {properties.map((p) => (
            <PropertyCard key={p.id} property={p} />
          ))}
        </div>
      )}

      {showUpload && <UploadModal onClose={() => setShowUpload(false)} onSuccess={refresh} />}
    </div>
  );
}
