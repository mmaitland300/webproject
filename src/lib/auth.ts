import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { prisma } from "@/lib/prisma";
import { isAdminAuthConfigured } from "@/lib/feature-config";
import { getAdminAuthEnv } from "@/lib/env";

const adminAuthConfigured = isAdminAuthConfigured();
const adminAuthEnv = adminAuthConfigured ? getAdminAuthEnv() : null;
const providers = adminAuthEnv
  ? [
      GitHub({
        clientId: adminAuthEnv.AUTH_GITHUB_ID,
        clientSecret: adminAuthEnv.AUTH_GITHUB_SECRET,
      }),
    ]
  : [];

export const { handlers, auth, signIn, signOut } = NextAuth({
  ...(adminAuthConfigured ? { adapter: PrismaAdapter(prisma) } : {}),
  providers,
  pages: {
    signIn: "/admin/login",
  },
  secret: adminAuthEnv?.AUTH_SECRET ?? "admin-auth-disabled",
  // Required by Auth.js when the request host must be trusted (avoids error=Configuration on some setups).
  // Local dev: set AUTH_URL=http://localhost:3000 in .env.local so OAuth does not use your production site URL.
  trustHost: true,
});
