// src/pages/NotFoundPage.tsx

import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <div className="min-h-screen bg-glome-off-white flex flex-col items-center justify-center text-center px-4">
      <p className="text-6xl mb-4">🏚️</p>
      <h1 className="text-3xl font-bold text-glome-navy mb-2">Page not found</h1>
      <p className="text-gray-400 mb-8">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <Link to="/" className="btn-primary">
        Go home
      </Link>
    </div>
  )
}
