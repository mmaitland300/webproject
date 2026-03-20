import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

const APEX_HOST = "mmaitland.dev";
const WWW_HOST = "www.mmaitland.dev";

export function middleware(request: NextRequest) {
  const host = request.headers.get("host");
  if (host !== WWW_HOST) return NextResponse.next();

  const url = request.nextUrl.clone();
  url.protocol = "https";
  url.host = APEX_HOST;
  return NextResponse.redirect(url, 308);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
