// src/pages/ViewerPage.tsx
import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api, Property, SemanticCategory, ObjectInstance } from '../lib/api';
import { ThreeViewer } from '../components/ThreeViewer';
import { Minimap } from '../components/Minimap';
import { SemanticObjectSidebar } from '../components/SemanticObjectSidebar';

export default function ViewerPage() {
  const { propertyId } = useParams<{ propertyId: string }>();
  const [property, setProperty] = useState<Property | null>(null);
  const [categories, setCategories] = useState<SemanticCategory[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [focusedInstance, setFocusedInstance] = useState<ObjectInstance | null>(null);
  const [loading, setLoading] = useState(true);
  const [cameraPos, setCameraPos] = useState({ x: 0, y: 1.6, z: 3, rotation: 0 });

  useEffect(() => {
    if (!propertyId) return;

    Promise.all([
      api.getProperty(propertyId),
      api.getPropertyObjects(propertyId).catch(() => ({ categories: [] })),
    ])
      .then(([p, objRes]) => {
        setProperty(p);
        setCategories(objRes.categories || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, [propertyId]);

  if (loading) {
    return (
      <div className="w-screen h-screen flex items-center justify-center bg-slate-950 text-white">
        <div className="w-10 h-10 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin" />
      </div>
    );
  }

  if (!property) {
    return (
      <div className="w-screen h-screen flex flex-col items-center justify-center bg-slate-950 text-white">
        <p className="text-xl mb-4">Property tour not found</p>
        <Link to="/dashboard" className="btn-primary text-sm">
          Return to Dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-slate-950">
      {/* Top Bar Header */}
      <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-6 py-4 bg-gradient-to-b from-slate-950/90 to-transparent pointer-events-none">
        <div className="flex items-center gap-4 pointer-events-auto">
          <Link
            to={`/properties/${property.id}`}
            className="px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/60 text-xs font-semibold text-white transition-colors"
          >
            ← Back
          </Link>
          <div>
            <h1 className="text-white font-bold text-sm tracking-wide">{property.address}</h1>
            <p className="text-slate-400 text-xs">Glome Interactive 3D Tour</p>
          </div>
        </div>

        <div className="flex items-center gap-2 pointer-events-auto">
          <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            ● 6-DoF Metric
          </span>
        </div>
      </div>

      {/* Semantic Object Sidebar */}
      <SemanticObjectSidebar
        categories={categories}
        selectedCategory={selectedCategory}
        onSelectCategory={setSelectedCategory}
        onFocusInstance={setFocusedInstance}
      />

      {/* 3D WebGL Canvas */}
      <ThreeViewer
        meshUrl={property.meshUrl}
        splatUrl={property.splatUrl}
        initialMode="mesh"
        categories={categories}
        selectedCategory={selectedCategory}
        focusedInstance={focusedInstance}
        onCameraUpdate={setCameraPos}
      />

      {/* 2D Minimap Overlay */}
      <Minimap cameraPos={cameraPos} />
    </div>
  );
}
