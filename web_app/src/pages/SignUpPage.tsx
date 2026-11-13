// src/pages/SignUpPage.tsx
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../lib/api';

export default function SignUpPage() {
  const navigate = useNavigate();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSignUp(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await api.signup(email.trim(), password, name.trim());
      navigate('/org/setup');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Sign up failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-glome-cream flex items-center justify-center px-4 selection:bg-glome-orange/20 selection:text-glome-orange-dark">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-glome-orange to-glome-orange-light flex items-center justify-center text-white shadow-sm font-bold text-sm">
              G
            </div>
            <span className="font-extrabold text-glome-charcoal text-2xl tracking-tight">Glome</span>
          </Link>
          <h1 className="text-2xl font-bold text-glome-charcoal mt-4 tracking-tight">Create your account</h1>
          <p className="text-glome-muted text-xs mt-1">Start generating 3D property twins</p>
        </div>

        <div className="liquid-glass p-8 space-y-5 rounded-3xl border border-glome-stone/60 shadow-glass-md bg-white/80 backdrop-blur-md">
          <form onSubmit={handleSignUp} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-glome-charcoal mb-1.5 uppercase tracking-wider">Full name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="input"
                placeholder="Jane Smith"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-glome-charcoal mb-1.5 uppercase tracking-wider">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input"
                placeholder="you@example.com"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-glome-charcoal mb-1.5 uppercase tracking-wider">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input"
                placeholder="Min. 8 characters"
                minLength={8}
                required
              />
            </div>

            {error && (
              <p className="text-red-600 text-xs bg-red-50 border border-red-200 px-3 py-2 rounded-xl">
                {error}
              </p>
            )}

            <button type="submit" disabled={loading} className="btn-primary w-full !py-3 shadow-md mt-2">
              {loading ? 'Creating account…' : 'Create Free Account'}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-glome-muted mt-6">
          Already have an account?{' '}
          <Link to="/sign-in" className="text-glome-orange font-semibold hover:underline">
            Sign in
          </Link>
          {' '}•{' '}
          <Link to="/" className="text-glome-muted hover:text-glome-charcoal hover:underline">
            Back to Agency
          </Link>
        </p>
      </div>
    </div>
  );
}
