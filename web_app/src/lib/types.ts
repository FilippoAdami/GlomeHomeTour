// src/lib/types.ts
// Shared Firestore document types for the Glome web app.

import { Timestamp } from 'firebase/firestore'

export type OrgRole = 'admin' | 'agent' | 'viewer'

export interface Organisation {
  id: string
  name: string
  plan: 'free' | 'pro'
  createdAt: Timestamp
}

export interface OrgMember {
  userId: string
  role: OrgRole
  displayName: string
  email: string
  joinedAt: Timestamp
}

export type PropertyStatus =
  | 'uploading'
  | 'processing'
  | 'ready'
  | 'error'

export interface Property {
  id: string
  orgId: string
  uploadedBy: string
  address: string
  description?: string
  createdAt: Timestamp
  status: PropertyStatus
  /** Firebase Storage download URLs — populated once pipeline completes */
  splatUrl?: string
  meshUrl?: string
  floorPlanUrl?: string
  manifestUrl?: string
  /** Archive upload path in Storage */
  archivePath?: string
}

export type PipelineStageStatus = 'queued' | 'running' | 'done' | 'error' | 'skipped'

export interface PipelineStage {
  name: string
  displayName: string
  status: PipelineStageStatus
  startedAt?: Timestamp
  completedAt?: Timestamp
  elapsedSeconds?: number
  error?: string
}

export interface PipelineJob {
  id: string
  propertyId: string
  startedAt: Timestamp
  completedAt?: Timestamp
  stages: PipelineStage[]
  error?: string
}

export interface ShareLink {
  id: string
  propertyId: string
  orgId: string
  createdBy: string
  requiresAccount: true
  expiresAt?: Timestamp
  createdAt: Timestamp
}

/** Invite document stored under organisations/{orgId}/invites/{inviteId} */
export interface OrgInvite {
  id: string
  email: string
  role: OrgRole
  invitedBy: string
  createdAt: Timestamp
  accepted: boolean
}
