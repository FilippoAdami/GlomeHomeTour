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
    <div className="flex h-screen overflow-hidden bg-glome-cream text-glome-charcoal">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-glome-charcoal flex flex-col border-r border-glome-charcoal-light/50">
        {/* Logo / Context */}
        <div className="px-6 py-5 border-b border-white/10">
          <div className="flex items-center justify-between">
            <span className="text-white font-extrabold text-xl tracking-tight flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-glome-orange inline-block" />
              Glome
            </span>
            <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-glome-orange/20 text-glome-orange border border-glome-orange/30">
              Home Tour
            </span>
          </div>
          {org && (
            <p className="text-white/50 text-xs mt-1.5 truncate font-medium">{org.name}</p>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1.5">
          {NAV_ITEMS.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                  isActive
                    ? 'bg-glome-orange text-white shadow-sm'
                    : 'text-white/70 hover:bg-white/10 hover:text-white'
                }`
              }
            >
              <span className="text-sm">{icon}</span>
              {label}
            </NavLink>
          ))}

          {/* Quick link back to Agency home */}
          <div className="pt-4 mt-4 border-t border-white/10 px-1">
            <NavLink
              to="/"
              className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs text-white/50 hover:text-white hover:bg-white/5 transition-colors"
            >
              <span>←</span>
              <span>Glome Agency Portal</span>
            </NavLink>
          </div>
        </nav>

        {/* User footer */}
        <div className="px-5 py-4 border-t border-white/10 bg-black/10">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-glome-orange to-glome-orange-light flex items-center justify-center text-white text-xs font-bold shadow-sm">
              {user?.displayName?.[0]?.toUpperCase() ?? user?.email?.[0]?.toUpperCase() ?? '?'}
            </div>
            <div className="min-w-0">
              <p className="text-white text-xs font-medium truncate">
                {user?.displayName ?? 'User'}
              </p>
              <p className="text-white/40 text-[11px] truncate">{user?.email}</p>
            </div>
          </div>
          <button
            onClick={handleSignOut}
            className="w-full text-left text-white/50 hover:text-glome-orange text-xs py-1 transition-colors flex items-center justify-between"
          >
            <span>Sign out</span>
            <span className="text-[10px]">↳</span>
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-glome-cream">
        <Outlet />
      </main>
    </div>
  );
}
