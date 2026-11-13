// src/components/NavigationBar.tsx
import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

export function NavigationBar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex justify-center px-4 pt-4 transition-all duration-300">
      <nav
        className={`w-full max-w-5xl flex items-center justify-between px-6 py-3 rounded-2xl transition-all duration-300 ${
          scrolled
            ? 'liquid-glass shadow-glass-md py-2.5 bg-white/80'
            : 'liquid-glass shadow-glass-sm bg-white/60'
        }`}
      >
        {/* Brand / Monogram */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-glome-orange to-glome-orange-light flex items-center justify-center text-white shadow-sm transition-transform group-hover:scale-105">
            <span className="font-extrabold text-sm tracking-tighter">G</span>
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-glome-charcoal text-base tracking-tight leading-tight">
              Glome
            </span>
            <span className="text-[10px] text-glome-muted tracking-wider uppercase font-medium">
              3D Reconstruction
            </span>
          </div>
        </Link>

        {/* Desktop Anchor Navigation */}
        <div className="hidden md:flex items-center gap-7 text-xs font-semibold text-glome-muted">
          <a
            href="#mission"
            className="hover:text-glome-charcoal transition-colors py-1"
          >
            Mission
          </a>
          <a
            href="#products"
            className="hover:text-glome-charcoal transition-colors py-1"
          >
            Products
          </a>
          <a
            href="#capabilities"
            className="hover:text-glome-charcoal transition-colors py-1"
          >
            Capabilities
          </a>
          <a
            href="#showcase"
            className="hover:text-glome-charcoal transition-colors py-1"
          >
            Showcase
          </a>
          <a
            href="#faq"
            className="hover:text-glome-charcoal transition-colors py-1"
          >
            FAQ
          </a>
        </div>

        {/* Right CTA */}
        <div className="hidden sm:flex items-center gap-3">
          <Link
            to="/sign-in"
            className="text-xs font-semibold text-glome-charcoal hover:text-glome-orange px-3 py-1.5 transition-colors"
          >
            Client Portal
          </Link>
          <Link
            to="/dashboard"
            className="btn-primary text-xs !py-2 !px-4 shadow-sm"
          >
            <span>Launch Home Tour</span>
            <span className="text-xs">→</span>
          </Link>
        </div>

        {/* Mobile menu button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 text-glome-charcoal hover:text-glome-orange focus:outline-none"
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </nav>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-20 left-4 right-4 liquid-glass shadow-glass-lg rounded-2xl p-5 flex flex-col gap-3 z-50">
          <a
            href="#mission"
            onClick={() => setMobileMenuOpen(false)}
            className="text-sm font-medium text-glome-charcoal py-2 border-b border-glome-stone/30"
          >
            Mission
          </a>
          <a
            href="#products"
            onClick={() => setMobileMenuOpen(false)}
            className="text-sm font-medium text-glome-charcoal py-2 border-b border-glome-stone/30"
          >
            Products
          </a>
          <a
            href="#capabilities"
            onClick={() => setMobileMenuOpen(false)}
            className="text-sm font-medium text-glome-charcoal py-2 border-b border-glome-stone/30"
          >
            Capabilities
          </a>
          <a
            href="#showcase"
            onClick={() => setMobileMenuOpen(false)}
            className="text-sm font-medium text-glome-charcoal py-2 border-b border-glome-stone/30"
          >
            Showcase
          </a>
          <a
            href="#faq"
            onClick={() => setMobileMenuOpen(false)}
            className="text-sm font-medium text-glome-charcoal py-2 border-b border-glome-stone/30"
          >
            FAQ
          </a>
          <div className="pt-2 flex flex-col gap-2">
            <Link
              to="/sign-in"
              onClick={() => setMobileMenuOpen(false)}
              className="btn-secondary text-xs text-center justify-center"
            >
              Sign In
            </Link>
            <Link
              to="/dashboard"
              onClick={() => setMobileMenuOpen(false)}
              className="btn-primary text-xs text-center justify-center"
            >
              Launch Home Tour
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
