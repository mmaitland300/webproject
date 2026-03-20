# Matt Maitland - Portfolio
<!-- deployment no-op: trigger Vercel rebuild -->

Personal portfolio and blog built with Next.js, TypeScript, and Tailwind CSS. Includes MDX blogging, a contact form with rate limiting, and an admin inbox with GitHub OAuth.

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
| `DATABASE_URL` | No | Neon PostgreSQL connection string (pooled). Required for waitlist persistence and admin auth. |
| `DIRECT_URL` | No | Neon PostgreSQL direct connection (for Prisma CLI) |
| `AUTH_SECRET` | No | Auth.js secret (`npx auth secret` to generate). Required for admin auth. |
| `AUTH_GITHUB_ID` | No | GitHub OAuth app client ID. Required for admin auth. |
| `AUTH_GITHUB_SECRET` | No | GitHub OAuth app client secret. Required for admin auth. |
| `ADMIN_GITHUB_IDS` | No | Comma-separated GitHub numeric user IDs for admin access |

## Database Setup (Optional)

The site works without a database. To enable the admin inbox and contact persistence:

1. Create a [Neon](https://neon.tech) PostgreSQL database.
2. Set `DATABASE_URL` and `DIRECT_URL` in `.env`.
3. Push the schema: `npx prisma db push`

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

## Operational Decisions

These are intentional tradeoffs built into the app. They are documented here so they are visible at the repo level, not just in code comments.

**Contact pipeline: email is source of truth, inbox is best-effort.**
The contact form sends email via Resend before attempting database persistence. If Prisma fails after a successful send, the user still gets a success response and the email is delivered. The admin inbox may miss that message. This is an intentional tradeoff: guaranteed delivery over guaranteed persistence.

**Public contact address is decoupled from delivery address.**
`contact@mmaitland.dev` (via ImprovMX forwarding) is what visitors see on the site. `CONTACT_TO_EMAIL` controls where form submissions actually go (currently the Gmail address directly). These are separate so the public-facing address can change without touching the app.

**Admin gating uses GitHub numeric user IDs, not email.**
`ADMIN_GITHUB_IDS` contains stable numeric IDs (e.g. `68873951`), not emails. GitHub user IDs do not change; email-based gating is unreliable because of privacy settings and address changes.

**Rate limiting is durable, not in-memory.**
The contact form uses Upstash Redis for rate limiting. In-memory state is not reliable across Vercel function instances. Upstash provides durable, globally consistent counters.

**Database and auth are optional, but they gate specific features.**
The site runs without `DATABASE_URL` or auth credentials. Without a database, the contact form still sends email, but the StringFlux waitlist and admin inbox are disabled. Admin access additionally requires `AUTH_SECRET`, `AUTH_GITHUB_ID`, and `AUTH_GITHUB_SECRET`.

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start development server |
| `npm run build` | Generate Prisma client and build for production |
| `npm start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm test` | Run unit and data-integrity tests (Vitest) |
