// src/components/LiquidGlassBadge.tsx
import React from 'react';

interface Props {
  variant: 'orange' | 'moss' | 'blueprint' | 'neutral';
  children: React.ReactNode;
  pulse?: boolean;
  className?: string;
}

export function LiquidGlassBadge({ variant, children, pulse = false, className = '' }: Props) {
  const styles = {
    orange: 'bg-glome-orange/10 text-glome-orange border-glome-orange/30 shadow-[0_2px_10px_rgba(217,93,40,0.15)]',
    moss: 'bg-glome-moss/10 text-glome-moss border-glome-moss/30 shadow-[0_2px_10px_rgba(68,107,75,0.15)]',
    blueprint: 'bg-glome-blueprint/10 text-glome-blueprint border-glome-blueprint/30 shadow-[0_2px_10px_rgba(30,84,148,0.15)]',
    neutral: 'bg-glome-charcoal/5 text-glome-muted border-glome-charcoal/15 shadow-none',
  };

  const dotStyles = {
    orange: 'bg-glome-orange',
    moss: 'bg-glome-moss',
    blueprint: 'bg-glome-blueprint',
    neutral: 'bg-glome-muted',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-semibold tracking-wide border backdrop-blur-md ${styles[variant]} ${className}`}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${dotStyles[variant]}`}
          />
          <span className={`relative inline-flex rounded-full h-2 w-2 ${dotStyles[variant]}`} />
        </span>
      )}
      {children}
    </span>
  );
}
