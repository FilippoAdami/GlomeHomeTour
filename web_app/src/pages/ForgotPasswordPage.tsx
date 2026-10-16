// src/pages/ForgotPasswordPage.tsx
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../lib/api';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [resetToken, setResetToken] = useState<string | null>(null);

  // Reset form state
  const [newPassword, setNewPassword] = useState('');
  const [resetSuccess, setResetSuccess] = useState(false);

  async function handleRequestToken(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await api.forgotPassword(email.trim());
      if (res.resetToken) {
        setResetToken(res.resetToken);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to request reset token.');
    } finally {
      setLoading(false);
    }
  }

  async function handleResetPassword(e: React.FormEvent) {
    e.preventDefault();
    if (!resetToken) return;
    setError('');
    setLoading(true);
    try {
      await api.resetPassword(resetToken, newPassword);
      setResetSuccess(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to reset password.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-glome-off-white flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Link to="/" className="inline-block">
            <span className="font-bold text-glome-navy text-2xl tracking-tight">Glome</span>
          </Link>
          <h1 className="text-2xl font-bold text-glome-navy mt-4">Reset password</h1>
          <p className="text-gray-500 text-sm mt-1">
            {resetSuccess
              ? 'Password successfully updated'
              : resetToken
              ? 'Enter your new password below'
              : 'Enter your email to receive a password reset token'}
          </p>
        </div>

        <div className="card p-8 space-y-5">
          {resetSuccess ? (
            <div className="text-center space-y-4">
              <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto text-xl">
                ✓
              </div>
              <p className="text-sm text-gray-600">
                Your password has been changed. You can now sign in with your new credentials.
              </p>
              <Link to="/sign-in" className="btn-primary w-full inline-block text-center">
                Sign in
              </Link>
            </div>
          ) : resetToken ? (
            <form onSubmit={handleResetPassword} className="space-y-4">
              <div className="bg-blue-50 border border-blue-200 p-3 rounded-lg text-xs text-blue-800">
                Reset Token validated for <b>{email}</b>.
              </div>

              <div>
                <label className="block text-sm font-medium text-glome-navy mb-1.5">New password</label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="input"
                  placeholder="Min. 8 characters"
                  minLength={8}
                  required
                />
              </div>

              {error && (
                <p className="text-red-500 text-sm bg-red-50 border border-red-200 px-3 py-2 rounded-lg">
                  {error}
                </p>
              )}

              <button type="submit" disabled={loading} className="btn-primary w-full">
                {loading ? 'Updating…' : 'Set new password'}
              </button>
            </form>
          ) : (
            <form onSubmit={handleRequestToken} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-glome-navy mb-1.5">Email address</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="input"
                  placeholder="you@example.com"
                  required
                />
              </div>

              {error && (
                <p className="text-red-500 text-sm bg-red-50 border border-red-200 px-3 py-2 rounded-lg">
                  {error}
                </p>
              )}

              <button type="submit" disabled={loading} className="btn-primary w-full">
                {loading ? 'Verifying…' : 'Request reset link'}
              </button>
            </form>
          )}
        </div>

        <p className="text-center text-sm text-gray-500 mt-6">
          Remembered your password?{' '}
          <Link to="/sign-in" className="text-glome-blue font-medium hover:underline">
            Back to sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
