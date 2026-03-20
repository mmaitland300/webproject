export function isWaitlistConfigured(
  env: NodeJS.ProcessEnv = process.env
): boolean {
  return Boolean(env.DATABASE_URL);
}

export function isAdminAuthConfigured(
  env: NodeJS.ProcessEnv = process.env
): boolean {
  return Boolean(
    env.DATABASE_URL &&
      env.AUTH_SECRET &&
      env.AUTH_GITHUB_ID &&
      env.AUTH_GITHUB_SECRET
  );
}
