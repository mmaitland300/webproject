import { isAdminAuthEnvConfigured, isWaitlistEnvConfigured } from "@/lib/env";

export function isWaitlistConfigured(
  env: NodeJS.ProcessEnv = process.env
): boolean {
  return isWaitlistEnvConfigured(env);
}

export function isAdminAuthConfigured(
  env: NodeJS.ProcessEnv = process.env
): boolean {
  return isAdminAuthEnvConfigured(env);
}
