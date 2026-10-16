// src/hooks/useProperties.ts
import { useState, useEffect } from 'react';
import { api, Property } from '../lib/api';

export function useProperties(orgId: string | null): {
  properties: Property[];
  loading: boolean;
  refresh: () => Promise<void>;
} {
  const [properties, setProperties] = useState<Property[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchProps = async () => {
    if (!orgId) {
      setProperties([]);
      setLoading(false);
      return;
    }
    try {
      const data = await api.getProperties();
      setProperties(data);
    } catch (err) {
      console.error('Error fetching properties', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProps();
    // Poll every 5 seconds for status updates while any property is processing
    const interval = setInterval(fetchProps, 5000);
    return () => clearInterval(interval);
  }, [orgId]);

  return { properties, loading, refresh: fetchProps };
}
