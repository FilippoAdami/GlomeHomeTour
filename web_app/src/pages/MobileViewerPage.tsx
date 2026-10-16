import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { ThreeViewer } from '../components/ThreeViewer';
import { API_BASE } from '../lib/api';

export default function MobileViewerPage() {
  const { propertyId = '' } = useParams();
  const [assets, setAssets] = useState<{ mesh: string; splat: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number;
    const checkAssets = async () => {
      const base = `${API_BASE}/static/outputs/${encodeURIComponent(propertyId)}`;
      let found = false;
      try {
        const [obj, meshPly, splat] = await Promise.all([
          fetch(`${base}/mesh_web.obj`, { method: 'HEAD' }),
          fetch(`${base}/mesh.ply`, { method: 'HEAD' }),
          fetch(`${base}/splat.ply`, { method: 'HEAD' }),
        ]);
        const meshUrl = obj.ok ? `${base}/mesh_web.obj` : meshPly.ok ? `${base}/mesh.ply` : null;
        found = meshUrl !== null && splat.ok;
        if (!cancelled && found && meshUrl) setAssets({ mesh: meshUrl, splat: `${base}/splat.ply` });
      } catch {
        // Pipeline may still be writing both output files.
      }
      if (!cancelled && !found) timer = window.setTimeout(checkAssets, 3000);
    };
    void checkAssets();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [propertyId]);

  if (!assets) {
    return <main className="flex h-screen items-center justify-center bg-slate-950 text-white">
      <p>Preparing your mesh and Gaussian splat…</p>
    </main>;
  }
  return <main className="h-screen w-screen bg-slate-950">
    <ThreeViewer meshUrl={assets.mesh} splatUrl={assets.splat} initialMode="splat" />
  </main>;
}
