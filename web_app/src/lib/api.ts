// src/lib/api.ts
// Local REST API client replacing Firebase SDK

export const API_BASE = import.meta.env.VITE_API_URL || `${window.location.protocol}//${window.location.hostname}:8000`;

export interface User {
  id: string;
  email: string;
  displayName: string;
}

export interface Organisation {
  id: string;
  name: string;
  plan: string;
  createdAt: string;
}

export interface OrgMember {
  userId: string;
  role: 'admin' | 'agent' | 'viewer';
  displayName: string;
  email: string;
  joinedAt: string;
}

export interface OrgInvite {
  id: string;
  email: string;
  role: string;
  accepted: boolean;
  createdAt: string;
}

export interface PipelineStage {
  name: string;
  displayName: string;
  status: 'queued' | 'running' | 'done' | 'error' | 'skipped';
  elapsedSeconds?: number;
  frames?: number;
  description?: string;
}

export interface PipelineJob {
  id: string;
  propertyId: string;
  status: 'queued' | 'running' | 'done' | 'error';
  startedAt: string;
  completedAt?: string;
  stages: PipelineStage[];
  error?: string;
}

export interface ObjectInstance {
  id: string;
  name: string;
  centroid: [number, number, number];
  dimensions?: string;
}

export interface SemanticCategory {
  id: string;
  label: string;
  icon: string;
  count: number;
  color: string;
  instances: ObjectInstance[];
}

export interface Property {
  id: string;
  orgId: string;
  uploadedBy: string;
  address: string;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  archivePath?: string;
  meshUrl?: string;
  splatUrl?: string;
  floorPlanUrl?: string;
  manifestUrl?: string;
  createdAt: string;
}

function getAuthHeader(): Record<string, string> {
  const token = localStorage.getItem('glome_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request(endpoint: string, options: RequestInit = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...getAuthHeader(),
    ...options.headers,
  };

  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let errMsg = `Request failed: ${res.status}`;
    try {
      const errData = await res.json();
      if (errData.detail) errMsg = errData.detail;
    } catch {}
    throw new Error(errMsg);
  }

  return res.json();
}

export const api = {
  // Auth
  async signup(email: string, password: string, displayName: string) {
    const data = await request('/api/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ email, password, display_name: displayName }),
    });
    localStorage.setItem('glome_token', data.token);
    localStorage.setItem('glome_user', JSON.stringify(data.user));
    return data;
  },

  async signin(email: string, password: string) {
    const data = await request('/api/auth/signin', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    localStorage.setItem('glome_token', data.token);
    localStorage.setItem('glome_user', JSON.stringify(data.user));
    return data;
  },

  async forgotPassword(email: string): Promise<{ status: string; message: string; resetToken?: string }> {
    return request('/api/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  async resetPassword(token: string, newPassword: string): Promise<{ status: string; message: string }> {
    return request('/api/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify({ token, new_password: newPassword }),
    });
  },

  signout() {
    localStorage.removeItem('glome_token');
    localStorage.removeItem('glome_user');
  },

  getCurrentUser(): User | null {
    const raw = localStorage.getItem('glome_user');
    return raw ? JSON.parse(raw) : null;
  },

  async fetchMe() {
    const data = await request('/api/auth/me');
    localStorage.setItem('glome_user', JSON.stringify(data.user));
    return data.user;
  },

  // Org
  async getOrg(): Promise<{ org: Organisation | null; member: OrgMember | null }> {
    return request('/api/org');
  },

  async createOrg(name: string): Promise<{ orgId: string; name: string }> {
    return request('/api/org', {
      method: 'POST',
      body: JSON.stringify({ name }),
    });
  },

  async getOrgMembers(orgId: string): Promise<OrgMember[]> {
    return request(`/api/org/${orgId}/members`);
  },

  async getOrgInvites(orgId: string): Promise<OrgInvite[]> {
    return request(`/api/org/${orgId}/invites`);
  },

  async inviteMember(orgId: string, email: string, role: string) {
    return request(`/api/org/${orgId}/invites`, {
      method: 'POST',
      body: JSON.stringify({ email, role }),
    });
  },

  // Properties
  async getProperties(): Promise<Property[]> {
    return request('/api/properties');
  },

  async getProperty(id: string): Promise<Property> {
    return request(`/api/properties/${id}`);
  },

  async uploadProperty(address: string, file: File, onProgress?: (pct: number) => void): Promise<{ propertyId: string }> {
    const token = localStorage.getItem('glome_token');
    const formData = new FormData();
    formData.append('address', address);
    formData.append('file', file);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_BASE}/api/properties/upload`);
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          const pct = Math.round((e.loaded / e.total) * 100);
          onProgress(pct);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || 'Upload failed'));
          } catch {
            reject(new Error(`Upload failed with status ${xhr.status}`));
          }
        }
      };

      xhr.onerror = () => reject(new Error('Network error during upload'));
      xhr.send(formData);
    });
  },

  async shareProperty(propertyId: string): Promise<{ shareId: string }> {
    return request(`/api/properties/${propertyId}/share`, { method: 'POST' });
  },

  async getPropertyPipeline(propertyId: string): Promise<{ job: PipelineJob | null }> {
    return request(`/api/properties/${propertyId}/pipeline`);
  },

  async getPropertyObjects(propertyId: string): Promise<{ categories: SemanticCategory[] }> {
    return request(`/api/properties/${propertyId}/objects`);
  },
};
