// src/lib/firestore.ts
// Typed helper functions for Firestore reads and writes.

import {
  collection,
  doc,
  addDoc,
  setDoc,
  updateDoc,
  getDoc,
  getDocs,
  onSnapshot,
  query,
  where,
  orderBy,
  serverTimestamp,
  Unsubscribe,
} from 'firebase/firestore'
import { db } from './firebase'
import type {
  Organisation,
  OrgMember,
  Property,
  PropertyStatus,
  PipelineJob,
  ShareLink,
  OrgInvite,
  OrgRole,
} from './types'

// ── Organisations ────────────────────────────────────────────────────────────

export async function createOrganisation(
  name: string,
  userId: string,
  userEmail: string,
  displayName: string,
): Promise<string> {
  const orgRef = await addDoc(collection(db, 'organisations'), {
    name,
    plan: 'free',
    createdAt: serverTimestamp(),
  })

  // Add the creator as admin member
  await setDoc(
    doc(db, 'organisations', orgRef.id, 'members', userId),
    {
      userId,
      role: 'admin' as OrgRole,
      displayName,
      email: userEmail,
      joinedAt: serverTimestamp(),
    },
  )

  return orgRef.id
}

export async function getUserOrg(
  userId: string,
): Promise<{ org: Organisation; member: OrgMember } | null> {
  // Find all orgs where the user is a member
  const orgsSnap = await getDocs(collection(db, 'organisations'))
  for (const orgDoc of orgsSnap.docs) {
    const memberDoc = await getDoc(
      doc(db, 'organisations', orgDoc.id, 'members', userId),
    )
    if (memberDoc.exists()) {
      return {
        org: { id: orgDoc.id, ...orgDoc.data() } as Organisation,
        member: memberDoc.data() as OrgMember,
      }
    }
  }
  return null
}

export function subscribeToOrgMembers(
  orgId: string,
  onChange: (members: OrgMember[]) => void,
): Unsubscribe {
  return onSnapshot(
    collection(db, 'organisations', orgId, 'members'),
    (snap) => onChange(snap.docs.map((d) => d.data() as OrgMember)),
  )
}

// ── Properties ───────────────────────────────────────────────────────────────

export async function createProperty(
  orgId: string,
  uploadedBy: string,
  address: string,
  archivePath: string,
): Promise<string> {
  const ref = await addDoc(collection(db, 'properties'), {
    orgId,
    uploadedBy,
    address,
    archivePath,
    status: 'uploading' as PropertyStatus,
    createdAt: serverTimestamp(),
  })
  return ref.id
}

export async function updatePropertyStatus(
  propertyId: string,
  status: PropertyStatus,
  extras?: Partial<Property>,
): Promise<void> {
  await updateDoc(doc(db, 'properties', propertyId), { status, ...extras })
}

export function subscribeToOrgProperties(
  orgId: string,
  onChange: (properties: Property[]) => void,
): Unsubscribe {
  const q = query(
    collection(db, 'properties'),
    where('orgId', '==', orgId),
    orderBy('createdAt', 'desc'),
  )
  return onSnapshot(q, (snap) =>
    onChange(snap.docs.map((d) => ({ id: d.id, ...d.data() }) as Property)),
  )
}

export function subscribeToProperty(
  propertyId: string,
  onChange: (property: Property | null) => void,
): Unsubscribe {
  return onSnapshot(doc(db, 'properties', propertyId), (snap) =>
    onChange(snap.exists() ? ({ id: snap.id, ...snap.data() } as Property) : null),
  )
}

// ── Pipeline Jobs ─────────────────────────────────────────────────────────────

export function subscribeToLatestJob(
  propertyId: string,
  onChange: (job: PipelineJob | null) => void,
): Unsubscribe {
  const q = query(
    collection(db, 'properties', propertyId, 'pipelineJobs'),
    orderBy('startedAt', 'desc'),
  )
  return onSnapshot(q, (snap) => {
    const first = snap.docs[0]
    onChange(first ? ({ id: first.id, ...first.data() } as PipelineJob) : null)
  })
}

// ── Share Links ───────────────────────────────────────────────────────────────

export async function createShareLink(
  propertyId: string,
  orgId: string,
  createdBy: string,
): Promise<string> {
  const ref = await addDoc(collection(db, 'shareLinks'), {
    propertyId,
    orgId,
    createdBy,
    requiresAccount: true,
    createdAt: serverTimestamp(),
  })
  return ref.id
}

export async function getShareLink(linkId: string): Promise<ShareLink | null> {
  const snap = await getDoc(doc(db, 'shareLinks', linkId))
  return snap.exists() ? ({ id: snap.id, ...snap.data() } as ShareLink) : null
}

// ── Org Invites ───────────────────────────────────────────────────────────────

export async function inviteMember(
  orgId: string,
  email: string,
  role: OrgRole,
  invitedBy: string,
): Promise<string> {
  const ref = await addDoc(
    collection(db, 'organisations', orgId, 'invites'),
    {
      email,
      role,
      invitedBy,
      accepted: false,
      createdAt: serverTimestamp(),
    },
  )
  return ref.id
}

export function subscribeToInvites(
  orgId: string,
  onChange: (invites: OrgInvite[]) => void,
): Unsubscribe {
  return onSnapshot(
    collection(db, 'organisations', orgId, 'invites'),
    (snap) =>
      onChange(snap.docs.map((d) => ({ id: d.id, ...d.data() }) as OrgInvite)),
  )
}
