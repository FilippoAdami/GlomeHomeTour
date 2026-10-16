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
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-glome-navy">Properties</h1>
          <p className="text-gray-400 text-sm mt-0.5">
            {properties.length} {properties.length === 1 ? 'property' : 'properties'}
          </p>
        </div>
        <button
          onClick={() => setShowUpload(true)}
          className="btn-primary"
        >
          + New scan
        </button>
      </div>

      {/* Content */}
      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="card p-6 h-52 animate-pulse bg-gray-100" />
          ))}
        </div>
      ) : properties.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="text-6xl mb-4">🏠</div>
          <h2 className="text-xl font-bold text-glome-navy mb-2">No properties yet</h2>
          <p className="text-gray-400 text-sm max-w-sm">
            Upload a scan archive from the Glome mobile app to generate your 3D property tour locally.
          </p>
          <button onClick={() => setShowUpload(true)} className="btn-primary mt-6">
            Upload your first scan
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
