"use server";

import { signIn, signOut } from "@/lib/auth";

export async function signInWithGitHub(redirectTo: string) {
  await signIn("github", { redirectTo });
}

export async function signOutUser(redirectTo: string) {
  await signOut({ redirectTo });
}
