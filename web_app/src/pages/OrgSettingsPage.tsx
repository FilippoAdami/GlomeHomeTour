// src/pages/OrgSettingsPage.tsx
import { useState, useEffect } from 'react';
import { useOrg } from '../hooks/useOrg';
import { api, OrgMember, OrgInvite } from '../lib/api';

export default function OrgSettingsPage() {
  const { org, member } = useOrg();
  const [members, setMembers] = useState<OrgMember[]>([]);
  const [invites, setInvites] = useState<OrgInvite[]>([]);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<'admin' | 'agent' | 'viewer'>('agent');
  const [inviting, setInviting] = useState(false);
  const [inviteError, setInviteError] = useState('');
  const [inviteSuccess, setInviteSuccess] = useState(false);

  const isAdmin = member?.role === 'admin';

  const loadData = () => {
    if (!org) return;
    api.getOrgMembers(org.id).then(setMembers).catch(console.error);
    api.getOrgInvites(org.id).then(setInvites).catch(console.error);
  };

  useEffect(() => {
    loadData();
  }, [org?.id]);

  async function handleInvite(e: React.FormEvent) {
    e.preventDefault();
    if (!org) return;
    setInviteError('');
    setInviting(true);
    try {
      await api.inviteMember(org.id, inviteEmail.trim(), inviteRole);
      setInviteEmail('');
      setInviteSuccess(true);
      loadData();
      setTimeout(() => setInviteSuccess(false), 3000);
    } catch (err: unknown) {
      setInviteError(err instanceof Error ? err.message : 'Failed to send invite.');
    } finally {
      setInviting(false);
    }
  }

  const ROLE_LABELS: Record<string, string> = {
    admin: 'Admin',
    agent: 'Agent',
    viewer: 'Viewer',
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-glome-navy">Organisation Settings</h1>
        {org && <p className="text-gray-400 text-sm mt-0.5">{org.name}</p>}
      </div>

      {/* Members */}
      <div className="card p-8">
        <h2 className="text-base font-semibold text-glome-navy mb-5">Team members</h2>
        <div className="space-y-3">
          {members.map((m) => (
            <div key={m.userId} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-glome-blue/10 text-glome-blue flex items-center justify-center text-xs font-bold">
                  {m.displayName?.[0]?.toUpperCase() ?? '?'}
                </div>
                <div>
                  <p className="text-sm font-medium text-glome-navy">{m.displayName}</p>
                  <p className="text-xs text-gray-400">{m.email}</p>
                </div>
              </div>
              <span className="text-xs font-medium text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">
                {ROLE_LABELS[m.role] || m.role}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Invite */}
      {isAdmin && (
        <div className="card p-8">
          <h2 className="text-base font-semibold text-glome-navy mb-5">Invite a team member</h2>
          <form onSubmit={handleInvite} className="space-y-4">
            <div className="flex gap-3">
              <input
                type="email"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                className="input flex-1"
                placeholder="colleague@example.com"
                required
              />
              <select
                value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value as 'admin' | 'agent' | 'viewer')}
                className="input w-32"
              >
                <option value="agent">Agent</option>
                <option value="viewer">Viewer</option>
                <option value="admin">Admin</option>
              </select>
              <button type="submit" disabled={inviting} className="btn-primary shrink-0">
                {inviting ? 'Sending…' : 'Invite'}
              </button>
            </div>
            {inviteError && (
              <p className="text-red-500 text-sm">{inviteError}</p>
            )}
            {inviteSuccess && (
              <p className="text-green-600 text-sm">✓ Invite recorded!</p>
            )}
          </form>

          {invites.filter((i) => !i.accepted).length > 0 && (
            <div className="mt-6">
              <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">
                Pending invites
              </p>
              <div className="space-y-2">
                {invites
                  .filter((i) => !i.accepted)
                  .map((inv) => (
                    <div
                      key={inv.id}
                      className="flex items-center justify-between text-sm py-1.5 border-b border-gray-50 last:border-0"
                    >
                      <span className="text-glome-navy">{inv.email}</span>
                      <span className="text-gray-400 text-xs">
                        {ROLE_LABELS[inv.role] || inv.role} · Pending
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
