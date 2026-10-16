// src/components/UploadModal.tsx
import { useState, useCallback } from 'react';
import { api } from '../lib/api';

interface Props {
  onClose: () => void;
  onSuccess?: () => void;
}

export function UploadModal({ onClose, onSuccess }: Props) {
  const [address, setAddress] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }, []);

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !address.trim()) return;
    setError('');

    try {
      setProgress(0);
      await api.uploadProperty(address.trim(), file, (pct) => {
        setProgress(pct);
      });
      if (onSuccess) onSuccess();
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed.');
      setProgress(null);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="card w-full max-w-lg p-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-bold text-glome-navy">New property scan</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">✕</button>
        </div>

        <form onSubmit={handleUpload} className="space-y-5">
          {/* Address */}
          <div>
            <label className="block text-sm font-medium text-glome-navy mb-1.5">
              Property address
            </label>
            <input
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              className="input"
              placeholder="123 Main Street, London"
              required
            />
          </div>

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
              dragging
                ? 'border-glome-blue bg-glome-blue/5'
                : 'border-gray-200 hover:border-glome-blue/50'
            }`}
          >
            {file ? (
              <div>
                <p className="text-sm font-medium text-glome-navy">{file.name}</p>
                <p className="text-xs text-gray-400 mt-1">{(file.size / 1e6).toFixed(1)} MB</p>
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  className="text-xs text-red-400 hover:text-red-600 mt-2"
                >
                  Remove
                </button>
              </div>
            ) : (
              <div>
                <p className="text-2xl mb-2">📦</p>
                <p className="text-sm font-medium text-glome-navy">Drop your scan archive here</p>
                <p className="text-xs text-gray-400 mt-1">or{' '}
                  <label className="text-glome-blue cursor-pointer hover:underline">
                    browse files
                    <input
                      type="file"
                      accept=".zip,.tar,.tar.gz"
                      className="sr-only"
                      onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    />
                  </label>
                </p>
                <p className="text-xs text-gray-300 mt-1">ZIP archive from the Glome mobile app</p>
              </div>
            )}
          </div>

          {/* Progress */}
          {progress !== null && (
            <div>
              <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span>Uploading to local server…</span>
                <span>{progress}%</span>
              </div>
              <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-glome-blue rounded-full transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {error && (
            <p className="text-red-500 text-sm bg-red-50 border border-red-200 px-3 py-2 rounded-lg">
              {error}
            </p>
          )}

          <div className="flex gap-3">
            <button type="button" onClick={onClose} className="btn-secondary flex-1">
              Cancel
            </button>
            <button
              type="submit"
              disabled={!file || !address.trim() || progress !== null}
              className="btn-primary flex-1"
            >
              Upload & process
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
