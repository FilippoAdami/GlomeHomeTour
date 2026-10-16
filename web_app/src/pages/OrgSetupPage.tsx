// src/pages/OrgSetupPage.tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';

export default function OrgSetupPage() {
  const navigate = useNavigate();

  const [orgName, setOrgName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!orgName.trim()) return;
    setError('');
    setLoading(true);
    try {
      await api.createOrg(orgName.trim());
      navigate('/dashboard');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create organisation.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-glome-off-white flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <span className="font-bold text-glome-navy text-2xl tracking-tight">Glome</span>
          <h1 className="text-2xl font-bold text-glome-navy mt-4">Set up your organisation</h1>
          <p className="text-gray-500 text-sm mt-1">
            This is your workspace where team members and property scans live.
          </p>
        </div>

        <div className="card p-8">
          <form onSubmit={handleCreate} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-glome-navy mb-1.5">
                Organisation name
              </label>
              <input
                type="text"
                value={orgName}
                onChange={(e) => setOrgName(e.target.value)}
                className="input"
                placeholder="e.g. Smith & Partners Real Estate"
                required
              />
            </div>

            {error && (
              <p className="text-red-500 text-sm bg-red-50 border border-red-200 px-3 py-2 rounded-lg">
                {error}
              </p>
            )}

            <button type="submit" disabled={loading || !orgName.trim()} className="btn-primary w-full">
              {loading ? 'Creating…' : 'Create organisation'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
