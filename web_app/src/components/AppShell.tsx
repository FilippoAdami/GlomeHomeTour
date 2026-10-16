// src/components/AppShell.tsx
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import { useAuth } from '../hooks/useAuth';
import { useOrg } from '../hooks/useOrg';

const NAV_ITEMS = [
  { to: '/dashboard', label: 'Properties', icon: '🏠' },
  { to: '/settings', label: 'Org Settings', icon: '⚙️' },
];

export function AppShell() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { org } = useOrg();

  function handleSignOut() {
    api.signout();
    navigate('/');
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 bg-glome-navy flex flex-col">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-white/10">
          <span className="text-white font-bold text-xl tracking-tight">Glome</span>
          {org && (
            <p className="text-white/50 text-xs mt-0.5 truncate">{org.name}</p>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-glome-blue text-white'
                    : 'text-white/70 hover:bg-white/10 hover:text-white'
                }`
              }
            >
              <span>{icon}</span>
              {label}
            </NavLink>
          ))}
        </nav>

        {/* User footer */}
        <div className="px-4 py-4 border-t border-white/10">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-8 h-8 rounded-full bg-glome-blue flex items-center justify-center text-white text-xs font-bold">
              {user?.displayName?.[0]?.toUpperCase() ?? user?.email?.[0]?.toUpperCase() ?? '?'}
            </div>
            <div className="min-w-0">
              <p className="text-white text-xs font-medium truncate">
                {user?.displayName ?? 'User'}
              </p>
              <p className="text-white/50 text-xs truncate">{user?.email}</p>
            </div>
          </div>
          <button
            onClick={handleSignOut}
            className="w-full text-left text-white/60 hover:text-white text-xs py-1 transition-colors"
          >
            Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-glome-off-white">
        <Outlet />
      </main>
    </div>
  );
}
