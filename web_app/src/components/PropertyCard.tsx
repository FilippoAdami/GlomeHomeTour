// src/components/PropertyCard.tsx
import { Link } from 'react-router-dom';
import type { Property } from '../lib/api';

const STATUS_CONFIG: Record<
  Property['status'],
  { label: string; className: string }
> = {
  uploading:  { label: 'Uploading',   className: 'bg-blue-100 text-blue-700' },
  processing: { label: 'Processing',  className: 'bg-yellow-100 text-yellow-700' },
  ready:      { label: 'Ready',       className: 'bg-green-100 text-green-700' },
  error:      { label: 'Error',       className: 'bg-red-100 text-red-700' },
};

interface Props {
  property: Property;
}

export function PropertyCard({ property }: Props) {
  const { label, className } = STATUS_CONFIG[property.status] || {
    label: property.status,
    className: 'bg-gray-100 text-gray-700',
  };

  const date = property.createdAt
    ? new Date(property.createdAt).toLocaleDateString('en-GB', {
        day: 'numeric', month: 'short', year: 'numeric',
      })
    : '—';

  return (
    <Link
      to={`/properties/${property.id}`}
      className="card p-5 flex flex-col gap-3 hover:shadow-glass-md hover:border-glome-orange/40 transition-all duration-200 group bg-white/80 backdrop-blur-md border border-glome-stone/60 rounded-2xl"
    >
      <div className="w-full h-36 rounded-xl bg-gradient-to-br from-glome-charcoal via-glome-charcoal-light to-glome-orange/40 flex items-center justify-center text-white/40 text-4xl shadow-inner group-hover:scale-[1.01] transition-transform">
        🏠
      </div>

      <div className="flex items-start justify-between gap-2 pt-1">
        <div className="min-w-0">
          <h3 className="font-bold text-glome-charcoal text-sm truncate group-hover:text-glome-orange transition-colors">
            {property.address}
          </h3>
          <p className="text-glome-muted text-xs mt-0.5">{date}</p>
        </div>
        <span className={`shrink-0 text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${className}`}>
          {label}
        </span>
      </div>
    </Link>
  );
}
