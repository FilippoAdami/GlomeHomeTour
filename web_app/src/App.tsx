// src/App.tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';

// Pages
import LandingPage from './pages/LandingPage';
import SignInPage from './pages/SignInPage';
import SignUpPage from './pages/SignUpPage';
import ForgotPasswordPage from './pages/ForgotPasswordPage';
import OrgSetupPage from './pages/OrgSetupPage';
import DashboardPage from './pages/DashboardPage';
import PropertyDetailPage from './pages/PropertyDetailPage';
import OrgSettingsPage from './pages/OrgSettingsPage';
import ViewerPage from './pages/ViewerPage';
import MobileViewerPage from './pages/MobileViewerPage';
import NotFoundPage from './pages/NotFoundPage';

// Layout
import { ProtectedRoute } from './components/ProtectedRoute';
import { AppShell } from './components/AppShell';

export default function App() {
  const { loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900">
        <div className="w-8 h-8 border-4 border-glome-blue/30 border-t-glome-blue rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        {/* Public */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/sign-in" element={<SignInPage />} />
        <Route path="/sign-up" element={<SignUpPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />

        {/* Public or Shared Tour Viewer */}
        <Route path="/tour/:shareId" element={<ViewerPage />} />
        <Route path="/mobile-viewer/:propertyId" element={<MobileViewerPage />} />

        {/* Auth-required routes */}
        <Route element={<ProtectedRoute />}>
          <Route path="/org/setup" element={<OrgSetupPage />} />
          <Route path="/viewer/:propertyId" element={<ViewerPage />} />

          {/* App Shell with Sidebar */}
          <Route element={<AppShell />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/properties/:propertyId" element={<PropertyDetailPage />} />
            <Route path="/settings" element={<OrgSettingsPage />} />
          </Route>
        </Route>

        {/* Fallback */}
        <Route path="/404" element={<NotFoundPage />} />
        <Route path="*" element={<Navigate to="/404" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
