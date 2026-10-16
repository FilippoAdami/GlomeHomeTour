// src/hooks/useOrg.ts
import { useState, useEffect } from 'react';
import { api, Organisation, OrgMember } from '../lib/api';
import { useAuth } from './useAuth';

interface OrgState {
  org: Organisation | null;
  member: OrgMember | null;
  loading: boolean;
  error: string | null;
}

export function useOrg(): OrgState {
  const { user } = useAuth();
  const [state, setState] = useState<OrgState>({
    org: null,
    member: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    if (!user) {
      setState({ org: null, member: null, loading: false, error: null });
      return;
    }

    setState((s) => ({ ...s, loading: true }));
    api.getOrg()
      .then((res) => {
        setState({
          org: res.org,
          member: res.member,
          loading: false,
          error: null,
        });
      })
      .catch((err) => {
        setState({ org: null, member: null, loading: false, error: String(err.message || err) });
      });
  }, [user]);

  return state;
}
