# backend/server/main.py
import os
import uuid
import json
import shutil
import asyncio
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr

from .db import init_db, get_connection
from .auth import hash_password, verify_password, create_access_token, get_current_user

# Setup storage directories
BACKEND_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BACKEND_DIR / "server_storage"
ARCHIVES_DIR = STORAGE_DIR / "archives"
OUTPUTS_DIR = STORAGE_DIR / "outputs"
ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize database schema
init_db()

app = FastAPI(title="Glome Home Tour API", version="1.0.0")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve outputs (mesh.ply, splat.ply, etc.) statically
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Also serve current_scene for existing demo scans if available
CURRENT_SCENE_DIR = BACKEND_DIR / "current_scene"
if CURRENT_SCENE_DIR.exists():
    app.mount("/static/current_scene", StaticFiles(directory=str(CURRENT_SCENE_DIR)), name="current_scene")


# Models
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class PasswordForgotRequest(BaseModel):
    email: EmailStr

class PasswordResetRequest(BaseModel):
    token: str
    new_password: str

class CreateOrgRequest(BaseModel):
    name: str

class InviteRequest(BaseModel):
    email: EmailStr
    role: str = "agent"


# Auth Routes
@app.post("/api/auth/signup")
def signup(req: SignUpRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")

    conn = get_connection()
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (req.email.lower(),)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    user_id = str(uuid.uuid4())
    pw_hash = hash_password(req.password)
    # Auto-generate email verification token
    verify_token = str(uuid.uuid4())

    conn.execute(
        "INSERT INTO users (id, email, password_hash, display_name, is_verified) VALUES (?, ?, ?, ?, 1)",
        (user_id, req.email.lower(), pw_hash, req.display_name.strip())
    )
    conn.execute(
        "INSERT INTO email_verifications (token, user_id, expires_at) VALUES (?, ?, datetime('now', '+24 hours'))",
        (verify_token, user_id)
    )
    conn.commit()
    conn.close()

    token = create_access_token(user_id, req.email.lower())
    return {
        "token": token,
        "user": {
            "id": user_id,
            "email": req.email.lower(),
            "displayName": req.display_name.strip(),
            "isVerified": True
        }
    }

@app.post("/api/auth/signin")
def signin(req: SignInRequest):
    conn = get_connection()
    user = conn.execute("SELECT id, email, password_hash, display_name, is_verified FROM users WHERE email = ?", (req.email.lower(),)).fetchone()
    conn.close()

    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    token = create_access_token(user["id"], user["email"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "displayName": user["display_name"],
            "isVerified": bool(user["is_verified"])
        }
    }

@app.post("/api/auth/forgot-password")
def forgot_password(req: PasswordForgotRequest):
    conn = get_connection()
    user = conn.execute("SELECT id, email FROM users WHERE email = ?", (req.email.lower(),)).fetchone()
    if not user:
        conn.close()
        # Security best practice: don't reveal whether user exists
        return {"status": "ok", "message": "If an account exists, a reset link has been dispatched."}

    reset_token = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO password_resets (token, user_id, expires_at) VALUES (?, ?, datetime('now', '+2 hours'))",
        (reset_token, user["id"])
    )
    conn.commit()
    conn.close()

    # In production, this emails the user. Locally, we return the reset URL directly for immediate use
    return {
        "status": "ok",
        "message": "Reset token generated",
        "resetToken": reset_token
    }

@app.post("/api/auth/reset-password")
def reset_password(req: PasswordResetRequest):
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")

    conn = get_connection()
    row = conn.execute("""
        SELECT user_id, expires_at, used
        FROM password_resets
        WHERE token = ? AND used = 0 AND expires_at > datetime('now')
    """, (req.token,)).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    new_hash = hash_password(req.new_password)
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, row["user_id"]))
    conn.execute("UPDATE password_resets SET used = 1 WHERE token = ?", (req.token,))
    conn.commit()
    conn.close()

    return {"status": "ok", "message": "Password successfully reset. You may now sign in."}

@app.get("/api/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return {"user": current_user}


# Organisation Routes
@app.get("/api/org")
def get_user_org(current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    row = conn.execute("""
        SELECT o.id, o.name, o.plan, o.created_at, om.role, om.joined_at
        FROM organisations o
        JOIN org_members om ON o.id = om.org_id
        WHERE om.user_id = ?
        LIMIT 1
    """, (current_user["id"],)).fetchone()
    conn.close()

    if not row:
        return {"org": None, "member": None}

    return {
        "org": {"id": row["id"], "name": row["name"], "plan": row["plan"], "createdAt": row["created_at"]},
        "member": {
            "userId": current_user["id"],
            "role": row["role"],
            "displayName": current_user["display_name"],
            "email": current_user["email"],
            "joinedAt": row["joined_at"]
        }
    }

@app.post("/api/org")
def create_org(req: CreateOrgRequest, current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    org_id = str(uuid.uuid4())
    conn.execute("INSERT INTO organisations (id, name, plan) VALUES (?, ?, 'free')", (org_id, req.name.strip()))
    conn.execute("INSERT INTO org_members (org_id, user_id, role) VALUES (?, ?, 'admin')", (org_id, current_user["id"]))
    conn.commit()
    conn.close()
    return {"orgId": org_id, "name": req.name}

@app.get("/api/org/{org_id}/members")
def get_org_members(org_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    members = conn.execute("""
        SELECT u.id, u.email, u.display_name, om.role, om.joined_at
        FROM org_members om
        JOIN users u ON om.user_id = u.id
        WHERE om.org_id = ?
    """, (org_id,)).fetchall()
    conn.close()
    return [
        {
            "userId": m["id"],
            "email": m["email"],
            "displayName": m["display_name"],
            "role": m["role"],
            "joinedAt": m["joined_at"]
        } for m in members
    ]

@app.get("/api/org/{org_id}/invites")
def get_org_invites(org_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    invites = conn.execute("SELECT id, email, role, accepted, created_at FROM org_invites WHERE org_id = ?", (org_id,)).fetchall()
    conn.close()
    return [
        {
            "id": inv["id"],
            "email": inv["email"],
            "role": inv["role"],
            "accepted": bool(inv["accepted"]),
            "createdAt": inv["created_at"]
        } for inv in invites
    ]

@app.post("/api/org/{org_id}/invites")
def invite_user(org_id: str, req: InviteRequest, current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    invite_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO org_invites (id, org_id, email, role, invited_by) VALUES (?, ?, ?, ?, ?)",
        (invite_id, org_id, req.email, req.role, current_user["id"])
    )
    conn.commit()
    conn.close()
    return {"inviteId": invite_id}


# Properties & Uploads
@app.get("/api/properties")
def list_properties(current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    # Get user's org
    mem = conn.execute("SELECT org_id FROM org_members WHERE user_id = ? LIMIT 1", (current_user["id"],)).fetchone()
    if not mem:
        conn.close()
        return []

    props = conn.execute("""
        SELECT id, org_id, uploaded_by, address, status, archive_path, mesh_url, splat_url, floor_plan_url, manifest_url, created_at
        FROM properties
        WHERE org_id = ?
        ORDER BY created_at DESC
    """, (mem["org_id"],)).fetchall()
    conn.close()

    return [
        {
            "id": p["id"],
            "orgId": p["org_id"],
            "uploadedBy": p["uploaded_by"],
            "address": p["address"],
            "status": p["status"],
            "archivePath": p["archive_path"],
            "meshUrl": p["mesh_url"],
            "splatUrl": p["splat_url"],
            "floorPlanUrl": p["floor_plan_url"],
            "manifestUrl": p["manifest_url"],
            "createdAt": p["created_at"]
        } for p in props
    ]

@app.get("/api/properties/{prop_id}")
def get_property(prop_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    p = conn.execute("SELECT * FROM properties WHERE id = ?", (prop_id,)).fetchone()
    conn.close()
    if not p:
        raise HTTPException(status_code=404, detail="Property not found")
    return {
        "id": p["id"],
        "orgId": p["org_id"],
        "uploadedBy": p["uploaded_by"],
        "address": p["address"],
        "status": p["status"],
        "archivePath": p["archive_path"],
        "meshUrl": p["mesh_url"],
        "splatUrl": p["splat_url"],
        "floorPlanUrl": p["floor_plan_url"],
        "manifestUrl": p["manifest_url"],
        "createdAt": p["created_at"]
    }


def run_pipeline_task(property_id: str, archive_path: str):
    """Background task running the real Glome reconstruction pipeline."""
    from .pipeline_runner import execute_pipeline
    conn = get_connection()
    conn.execute("UPDATE properties SET status = 'processing' WHERE id = ?", (property_id,))
    conn.commit()
    conn.close()

    execute_pipeline(property_id, archive_path)


@app.post("/api/properties/upload")
async def upload_property(
    background_tasks: BackgroundTasks,
    address: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    conn = get_connection()
    mem = conn.execute("SELECT org_id FROM org_members WHERE user_id = ? LIMIT 1", (current_user["id"],)).fetchone()
    if not mem:
        conn.close()
        raise HTTPException(status_code=400, detail="User must belong to an organisation")

    property_id = str(uuid.uuid4())
    dest_filename = f"{property_id}_{file.filename}"
    file_path = ARCHIVES_DIR / dest_filename

    # Save file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conn.execute("""
        INSERT INTO properties (id, org_id, uploaded_by, address, status, archive_path)
        VALUES (?, ?, ?, ?, 'uploading', ?)
    """, (property_id, mem["org_id"], current_user["id"], address.strip(), str(file_path)))
    conn.commit()
    conn.close()

    # Launch processing in background
    background_tasks.add_task(run_pipeline_task, property_id, str(file_path))

    return {"propertyId": property_id, "status": "processing"}


@app.post("/api/properties/{prop_id}/share")
def create_share(prop_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    prop = conn.execute("SELECT org_id FROM properties WHERE id = ?", (prop_id,)).fetchone()
    if not prop:
        conn.close()
        raise HTTPException(status_code=404, detail="Property not found")

    share_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO share_links (id, property_id, org_id, created_by) VALUES (?, ?, ?, ?)",
        (share_id, prop_id, prop["org_id"], current_user["id"])
    )
    conn.commit()
    conn.close()
    return {"shareId": share_id}


@app.get("/api/properties/{prop_id}/pipeline")
def get_property_pipeline(prop_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    job = conn.execute("""
        SELECT id, property_id, status, started_at, completed_at, stages_json, error
        FROM pipeline_jobs
        WHERE property_id = ?
        ORDER BY started_at DESC
        LIMIT 1
    """, (prop_id,)).fetchone()
    conn.close()

    if not job:
        return {"job": None}

    stages = json.loads(job["stages_json"]) if job["stages_json"] else []
    return {
        "job": {
            "id": job["id"],
            "propertyId": job["property_id"],
            "status": job["status"],
            "startedAt": job["started_at"],
            "completedAt": job["completed_at"],
            "stages": stages,
            "error": job["error"]
        }
    }


@app.get("/api/properties/{prop_id}/objects")
def get_property_objects(prop_id: str, current_user: dict = Depends(get_current_user)):
    # Curated semantic objects detected in the bedroom scene
    # When fully integrated into the pipeline, this will read from objects_manifest.json
    return {
        "categories": [
            {
                "id": "bed",
                "label": "Double Bed",
                "icon": "🛏️",
                "count": 1,
                "color": "#3B82F6",
                "instances": [
                    {
                        "id": "inst_bed_1",
                        "name": "King Double Bed",
                        "centroid": [0.15, 0.45, -0.60],
                        "dimensions": "2.0m × 1.8m × 0.9m"
                    }
                ]
            },
            {
                "id": "nightstand",
                "label": "Bedside Nightstands",
                "icon": "🚪",
                "count": 2,
                "color": "#10B981",
                "instances": [
                    {
                        "id": "inst_ns_left",
                        "name": "Left Nightstand",
                        "centroid": [-1.05, 0.50, -0.75],
                        "dimensions": "0.5m × 0.4m × 0.6m"
                    },
                    {
                        "id": "inst_ns_right",
                        "name": "Right Nightstand",
                        "centroid": [1.35, 0.50, -0.75],
                        "dimensions": "0.5m × 0.4m × 0.6m"
                    }
                ]
            },
            {
                "id": "wardrobe",
                "label": "Wardrobe & Closets",
                "icon": "🗄️",
                "count": 1,
                "color": "#F59E0B",
                "instances": [
                    {
                        "id": "inst_wardrobe_1",
                        "name": "Main Wall Wardrobe",
                        "centroid": [-1.80, 1.10, 0.85],
                        "dimensions": "2.2m × 0.6m × 2.1m"
                    }
                ]
            },
            {
                "id": "lighting",
                "label": "Lamps & Lighting",
                "icon": "💡",
                "count": 2,
                "color": "#EC4899",
                "instances": [
                    {
                        "id": "inst_lamp_1",
                        "name": "Bedside Reading Lamp",
                        "centroid": [-1.05, 0.85, -0.75],
                        "dimensions": "0.2m × 0.2m × 0.4m"
                    },
                    {
                        "id": "inst_lamp_2",
                        "name": "Ceiling Fixture",
                        "centroid": [0.00, 2.40, 0.00],
                        "dimensions": "0.4m × 0.4m × 0.3m"
                    }
                ]
            },
            {
                "id": "mirror",
                "label": "Wall Mirror",
                "icon": "🪞",
                "count": 1,
                "color": "#8B5CF6",
                "instances": [
                    {
                        "id": "inst_mirror_1",
                        "name": "Full-Length Mirror",
                        "centroid": [1.70, 1.30, 0.40],
                        "dimensions": "0.6m × 0.05m × 1.5m"
                    }
                ]
            }
        ]
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Glome Local API"}
