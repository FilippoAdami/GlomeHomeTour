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
      className="card p-6 flex flex-col gap-3 hover:shadow-md hover:border-glome-blue/30 transition-all duration-150 group"
    >
      <div className="w-full h-36 rounded-xl bg-gradient-to-br from-glome-navy to-glome-blue/60 flex items-center justify-center text-white/30 text-4xl">
        🏠
      </div>

      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="font-semibold text-glome-navy text-sm truncate group-hover:text-glome-blue transition-colors">
            {property.address}
          </h3>
          <p className="text-gray-400 text-xs mt-0.5">{date}</p>
        </div>
        <span className={`shrink-0 text-xs font-medium px-2 py-0.5 rounded-full ${className}`}>
          {label}
        </span>
      </div>
    </Link>
  );
}
