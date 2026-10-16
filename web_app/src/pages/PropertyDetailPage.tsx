// src/pages/PropertyDetailPage.tsx
import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api, Property, PipelineJob } from '../lib/api';
import { PipelineTimeline } from '../components/PipelineTimeline';

const STATUS_STEPS: Array<{ key: Property['status']; label: string }> = [
  { key: 'uploading',  label: 'Uploading archive' },
  { key: 'processing', label: 'Running pipeline' },
  { key: 'ready',      label: 'Ready' },
];

export default function PropertyDetailPage() {
  const { propertyId } = useParams<{ propertyId: string }>();
  const [property, setProperty] = useState<Property | null>(null);
  const [pipelineJob, setPipelineJob] = useState<PipelineJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [shareLink, setShareLink] = useState<string | null>(null);
  const [copyDone, setCopyDone] = useState(false);

  const fetchPropertyAndPipeline = () => {
    if (!propertyId) return;
    Promise.all([
      api.getProperty(propertyId),
      api.getPropertyPipeline(propertyId).catch(() => ({ job: null }))
    ])
      .then(([p, pipeRes]) => {
        setProperty(p);
        setPipelineJob(pipeRes.job);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchPropertyAndPipeline();
    const interval = setInterval(fetchPropertyAndPipeline, 3000);
    return () => clearInterval(interval);
  }, [propertyId]);

  async function handleShare() {
    if (!property) return;
    try {
      const { shareId } = await api.shareProperty(property.id);
      const url = `${window.location.origin}/tour/${shareId}`;
      setShareLink(url);
      await navigator.clipboard.writeText(url).catch(() => undefined);
      setCopyDone(true);
      setTimeout(() => setCopyDone(false), 2000);
    } catch (err) {
      console.error(err);
    }
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-10">
        <div className="h-8 w-64 bg-gray-200 rounded animate-pulse mb-4" />
        <div className="card p-8 h-64 animate-pulse bg-gray-100" />
      </div>
    );
  }

  if (!property) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-10 text-center">
        <p className="text-gray-400">Property not found.</p>
        <Link to="/dashboard" className="btn-primary mt-4 inline-flex">
          Back to dashboard
        </Link>
      </div>
    );
  }

  const currentStepIndex = STATUS_STEPS.findIndex((s) => s.key === property.status);

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <Link to="/dashboard" className="hover:text-glome-blue transition-colors">
          Properties
        </Link>
        <span>/</span>
        <span className="text-glome-navy font-medium truncate">{property.address}</span>
      </div>

      {/* Header card */}
      <div className="card p-8 flex items-start justify-between gap-6">
        <div>
          <h1 className="text-2xl font-bold text-glome-navy">{property.address}</h1>
          <p className="text-gray-400 text-sm mt-1">
            Uploaded {property.createdAt ? new Date(property.createdAt).toLocaleDateString('en-GB', {
              day: 'numeric', month: 'long', year: 'numeric',
            }) : '—'}
          </p>
        </div>

        {property.status === 'ready' && (
          <div className="flex flex-col items-end gap-2 shrink-0">
            <Link
              to={`/viewer/${property.id}`}
              className="btn-primary"
            >
              🌐 Open 3D Tour
            </Link>
            <button onClick={handleShare} className="btn-secondary text-sm">
              {copyDone ? '✓ Copied!' : '🔗 Share link'}
            </button>
            {shareLink && (
              <p className="text-xs text-gray-400 max-w-xs truncate">{shareLink}</p>
            )}
          </div>
        )}
      </div>

      {/* Pipeline status */}
      <div className="card p-8">
        <h2 className="text-base font-semibold text-glome-navy mb-6">Processing status</h2>

        {property.status === 'error' ? (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-red-700 font-medium text-sm">Pipeline error</p>
            <p className="text-red-500 text-xs mt-1">
              Something went wrong during processing. Please re-upload or check server logs.
            </p>
          </div>
        ) : (
          <div className="flex items-center gap-0">
            {STATUS_STEPS.map((step, i) => {
              const done = i < currentStepIndex || property.status === 'ready';
              const active = i === currentStepIndex && property.status !== 'ready';
              return (
                <div key={step.key} className="flex items-center flex-1">
                  <div className="flex flex-col items-center">
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-colors ${
                        done
                          ? 'bg-green-500 text-white'
                          : active
                          ? 'bg-glome-blue text-white animate-pulse'
                          : 'bg-gray-100 text-gray-400'
                      }`}
                    >
                      {done ? '✓' : i + 1}
                    </div>
                    <p className={`text-xs mt-2 text-center max-w-[80px] ${
                      active ? 'text-glome-blue font-medium' : done ? 'text-gray-500' : 'text-gray-300'
                    }`}>
                      {step.label}
                    </p>
                  </div>
                  {i < STATUS_STEPS.length - 1 && (
                    <div className={`flex-1 h-0.5 mx-2 mb-5 transition-colors ${done ? 'bg-green-400' : 'bg-gray-200'}`} />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Detailed Pipeline Stage Breakdown */}
      <PipelineTimeline job={pipelineJob} overallStatus={property.status} />
    </div>
  );
}
