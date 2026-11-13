// src/pages/SignInPage.tsx
import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { api } from '../lib/api';

export default function SignInPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? '/dashboard';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSignIn(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await api.signin(email.trim(), password);
      // Check org status
      const orgData = await api.getOrg();
      if (!orgData.org) {
        navigate('/org/setup', { replace: true });
      } else {
        navigate(from, { replace: true });
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Sign in failed. Please check credentials.');
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
          <h1 className="text-2xl font-bold text-glome-charcoal mt-4 tracking-tight">Welcome back</h1>
          <p className="text-glome-muted text-xs mt-1">Sign in to your Glome 3D workspace</p>
        </div>

        <div className="liquid-glass p-8 space-y-5 rounded-3xl border border-glome-stone/60 shadow-glass-md bg-white/80 backdrop-blur-md">
          <form onSubmit={handleSignIn} className="space-y-4">
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
              <div className="flex items-center justify-between mb-1.5">
                <label className="block text-xs font-semibold text-glome-charcoal uppercase tracking-wider">Password</label>
                <Link to="/forgot-password" className="text-xs text-glome-orange hover:underline font-medium">
                  Forgot password?
                </Link>
              </div>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input"
                placeholder="••••••••"
                required
              />
            </div>

            {error && (
              <p className="text-red-600 text-xs bg-red-50 border border-red-200 px-3 py-2 rounded-xl">
                {error}
              </p>
            )}

            <button type="submit" disabled={loading} className="btn-primary w-full !py-3 shadow-md mt-2">
              {loading ? 'Signing in…' : 'Sign in to Workspace'}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-glome-muted mt-6">
          Don't have an account?{' '}
          <Link to="/sign-up" className="text-glome-orange font-semibold hover:underline">
            Create an account
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
