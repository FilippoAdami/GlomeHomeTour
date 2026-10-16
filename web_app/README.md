# Glome Web App

React + TypeScript + Vite SaaS dashboard for managing and sharing Glome 3D property tours.

## Stack

- **Framework**: React 18 + TypeScript (Vite)
- **Styling**: Tailwind CSS v3
- **Routing**: React Router v6
- **Backend**: Firebase (Auth + Firestore + Storage + Hosting)

## Getting started

### 1. Clone and install

```bash
cd web_app
npm install
```

### 2. Configure Firebase

Copy the env template and fill in your Firebase project values:

```bash
cp .env.example .env.local
# Edit .env.local with your Firebase config
```

You can find these values in the Firebase Console → Project Settings → Your apps → Web app.

### 3. Run in development

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### 4. Build for production

```bash
npm run build
```

### 5. Deploy to Firebase Hosting

```bash
npx -y firebase-tools@latest deploy --only hosting
```

## Project structure

```
web_app/
├── src/
│   ├── lib/
│   │   ├── firebase.ts       # Firebase app init (auth, db, storage)
│   │   ├── firestore.ts      # Typed Firestore helpers
│   │   └── types.ts          # Shared TypeScript types
│   ├── hooks/
│   │   ├── useAuth.ts        # Firebase auth state
│   │   ├── useOrg.ts         # Current user's organisation
│   │   └── useProperties.ts  # Real-time property list
│   ├── components/
│   │   ├── AppShell.tsx      # Sidebar + topbar layout
│   │   ├── ProtectedRoute.tsx
│   │   ├── PropertyCard.tsx
│   │   └── UploadModal.tsx   # Drag-and-drop archive upload
│   ├── pages/
│   │   ├── LandingPage.tsx
│   │   ├── SignInPage.tsx
│   │   ├── SignUpPage.tsx
│   │   ├── OrgSetupPage.tsx
│   │   ├── DashboardPage.tsx
│   │   ├── PropertyDetailPage.tsx
│   │   ├── OrgSettingsPage.tsx
│   │   └── NotFoundPage.tsx
│   ├── App.tsx               # Root router
│   ├── main.tsx
│   └── index.css             # Tailwind base + component layer
├── firebase.json             # Firebase project config
├── firestore.rules           # Firestore security rules
├── storage.rules             # Storage security rules
├── tailwind.config.js
├── vite.config.ts
└── .env.example              # Environment variable template
```

## Firebase setup

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Enable **Authentication** (Email/Password + Google)
3. Create a **Firestore** database
4. Create a **Storage** bucket
5. Register a **Web app** and copy the config to `.env.local`
6. Deploy auth config: `npx -y firebase-tools@latest deploy --only auth`
7. Deploy rules: `npx -y firebase-tools@latest deploy --only firestore:rules,storage`

## Phase roadmap

| Phase | Status |
|---|---|
| Phase 1: Auth + Dashboard | ✅ Complete |
| Phase 2: 3D Viewer (splat + mesh) | 🔜 Next |
| Phase 3: Pipeline Job Tracker | 🔜 Planned |
| Phase 4: Semantic Object Browser | 🔜 Planned |
