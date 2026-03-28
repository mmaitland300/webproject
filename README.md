# Matt Maitland - Portfolio

Portfolio site and engineering case-study repo for mmaitland.dev. Built with Next.js, TypeScript, and Tailwind CSS, with MDX blogging, a rate-limited contact pipeline, and optional admin inbox/auth features behind explicit configuration.

## What This Repo Demonstrates

- Structured content/data modeling for projects, resume content, and MDX posts
- Typed environment/config handling with optional-feature gating
- Operational safeguards (validation, rate limiting, and graceful degraded behavior)
- CI-backed quality checks (lint, tests, build, and route smoke coverage)

## Tech Stack

- **Next.js 16** (App Router, Server Components, Server Actions)
- **TypeScript**
- **Tailwind CSS v4** + **shadcn/ui**
- **Framer Motion** for animations
- **MDX** via `next-mdx-remote` for blog posts
- **Prisma** + PostgreSQL (Neon) for persistence
- **Auth.js v5** (next-auth) with GitHub OAuth
- **Resend** for contact form emails
- **Upstash Redis** for server-side rate limiting

## Getting Started

```powershell
git clone https://github.com/mmaitland300/webproject.git
cd webproject
Copy-Item .env.example .env   # PowerShell; then fill in values (see below)
npm install            # also runs prisma generate via postinstall
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment Variables

Copy `.env.example` and fill in the values. The site runs without the optional variables, but some features will be disabled.

| Variable | Required | Purpose |
|---|---|---|
| `NEXT_PUBLIC_SITE_URL` | Yes | Base URL for metadata, sitemap, OG images |
| `RESEND_API_KEY` | Yes | Resend API key for contact form delivery |
| `CONTACT_FROM_EMAIL` | Yes | Sender address for contact emails |
| `CONTACT_TO_EMAIL` | Yes | Recipient address for contact emails |
| `UPSTASH_REDIS_REST_URL` | Yes | Upstash Redis URL for rate limiting |
| `UPSTASH_REDIS_REST_TOKEN` | Yes | Upstash Redis token |
| `DATABASE_URL` | Yes† | Placeholder or Neon pooled URL. Required for `prisma generate` (see note below). Runtime DB features need a real Neon URL. |
| `DIRECT_URL` | Yes† | Placeholder or Neon direct URL for Prisma CLI migrations. Same note as `DATABASE_URL`. |
| `AUTH_SECRET` | No | Auth.js secret (`npx auth secret` to generate). Required for admin auth. |
| `AUTH_GITHUB_ID` | No | GitHub OAuth app client ID. Required for admin auth. |
| `AUTH_GITHUB_SECRET` | No | GitHub OAuth app client secret. Required for admin auth. |
| `ADMIN_GITHUB_IDS` | No | Comma-separated GitHub numeric user IDs for admin access |

† **Prisma tooling:** `prisma.config.ts` reads `DATABASE_URL` and `DIRECT_URL` via `env()`, so they must exist in `.env` for `npm install` (postinstall `prisma generate`) and `npm run build`. Copy the syntactically valid placeholders from `.env.example` until you point them at Neon; no Postgres process is required on your machine for generation or production build.

## Database Setup (Optional)

The site works without a database. To enable the admin inbox and contact persistence:

1. Create a [Neon](https://neon.tech) PostgreSQL database.
2. Replace the Prisma placeholders in `.env` with your Neon pooled (`DATABASE_URL`) and direct (`DIRECT_URL`) URLs.
3. Apply the schema: `npx prisma migrate deploy` (uses `prisma/migrations`). For a throwaway local database you can use `npx prisma db push` instead.

For hosted environments that use migration history, apply pending migrations during deploy or release with `npm run db:migrate:deploy` against your real database (not part of `npm run build`).

Note: the contact flow guarantees email delivery first. Database persistence for the admin inbox runs after a successful email send and is best-effort.

## Auth Setup (Optional)

To enable the admin dashboard at `/admin`:

1. Create a [GitHub OAuth App](https://github.com/settings/developers) (callback URL: `http://localhost:3000/api/auth/callback/github`).
2. Set `AUTH_SECRET`, `AUTH_GITHUB_ID`, and `AUTH_GITHUB_SECRET` in `.env`.
3. Set `ADMIN_GITHUB_IDS` to your GitHub numeric user ID (find it at `https://api.github.com/users/YOUR_USERNAME`).

## Project Structure

```
src/
  app/            # Next.js App Router pages and layouts
  actions/        # Server Actions (contact form, inbox mutations)
  components/     # UI and section components
  content/        # Blog posts (MDX) and project/resume data
  lib/            # Utilities (MDX, Prisma, auth, admin helpers)
  generated/      # Prisma client (auto-generated, gitignored)
prisma/           # Prisma schema
public/           # Static assets, game embeds, resume PDF
```

## Content Management

**Blog posts** live in `src/content/blog/` as `.mdx` files with YAML frontmatter:

```yaml
---
title: "Post Title"
description: "Short description"
date: "2026-03-18"
tags: ["Next.js", "TypeScript"]
published: true
---
```

Set `published: false` to keep a post as a draft (hidden from listings and direct URL access).

**Projects** are defined in `src/content/projects.ts`. Each entry has a `category` of `"featured"` or `"experiment"`.

**Resume data** is centralized in `src/content/resume.ts` and consumed by both the `/about` and `/resume` pages.

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start development server |
| `npm run build` | Generate Prisma client and build for production (no DB connection) |
| `npm run db:migrate:deploy` | Apply Prisma migrations to the target database (run at deploy/release) |
| `npm start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm test` | Run unit and data-integrity tests (Vitest) |
