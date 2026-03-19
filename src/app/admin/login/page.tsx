import { auth, signIn } from "@/lib/auth";
import { redirect } from "next/navigation";
import { Github, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { isAdmin } from "@/lib/admin";

export default async function AdminLoginPage() {
  const session = await auth();

  if (session) {
    const authorized = await isAdmin();
    if (authorized) {
      redirect("/admin/inbox");
    }

    return (
      <div className="py-32 flex items-center justify-center min-h-[60vh]">
        <div className="text-center space-y-6 max-w-md">
          <div className="mx-auto w-fit p-3 rounded-full bg-destructive/10">
            <ShieldAlert className="h-8 w-8 text-destructive" />
          </div>
          <h1 className="text-3xl font-bold">Access Denied</h1>
          <p className="text-muted-foreground">
            You&apos;re signed in as{" "}
            <span className="text-foreground font-medium">
              {session.user?.email ?? session.user?.name ?? "unknown"}
            </span>
            , but this account is not authorized to access the admin area.
          </p>
          <form
            action={async () => {
              "use server";
              const { signOut } = await import("@/lib/auth");
              await signOut({ redirectTo: "/" });
            }}
          >
            <Button type="submit" variant="outline">
              Sign out
            </Button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="py-32 flex items-center justify-center min-h-[60vh]">
      <div className="text-center space-y-6">
        <h1 className="text-3xl font-bold">Admin Login</h1>
        <p className="text-muted-foreground">
          Sign in with GitHub to access the admin dashboard.
        </p>
        <form
          action={async () => {
            "use server";
            await signIn("github", { redirectTo: "/admin/inbox" });
          }}
        >
          <Button
            type="submit"
            className="bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white border-0"
          >
            <Github className="mr-2 h-4 w-4" />
            Sign in with GitHub
          </Button>
        </form>
      </div>
    </div>
  );
}
