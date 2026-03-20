import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { prisma } from "@/lib/prisma";
import { isAdminAuthConfigured } from "@/lib/feature-config";

const adminAuthConfigured = isAdminAuthConfigured();

export const { handlers, auth, signIn, signOut } = NextAuth({
  ...(adminAuthConfigured ? { adapter: PrismaAdapter(prisma) } : {}),
  providers: adminAuthConfigured
    ? [
        GitHub({
          clientId: process.env.AUTH_GITHUB_ID!,
          clientSecret: process.env.AUTH_GITHUB_SECRET!,
        }),
      ]
    : [],
  pages: {
    signIn: "/admin/login",
  },
  secret: process.env.AUTH_SECRET ?? "admin-auth-disabled",
});
