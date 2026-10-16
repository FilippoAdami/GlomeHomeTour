// src/hooks/useAuth.ts
import { useState, useEffect } from 'react';
import { api, User } from '../lib/api';

interface AuthState {
  user: User | null;
  loading: boolean;
}

export function useAuth(): AuthState {
  const [state, setState] = useState<AuthState>({
    user: api.getCurrentUser(),
    loading: true,
  });

  useEffect(() => {
    const token = localStorage.getItem('glome_token');
    if (!token) {
      setState({ user: null, loading: false });
      return;
    }

    api.fetchMe()
      .then((user) => {
        setState({ user, loading: false });
      })
      .catch(() => {
        api.signout();
        setState({ user: null, loading: false });
      });
  }, []);

  return state;
}
