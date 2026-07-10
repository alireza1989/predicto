import { NextRequest, NextResponse } from "next/server";

// Demo gate — not a security boundary. The token is published in the repo
// README so anyone who can see the repo can view the dashboard; it only
// keeps random visitors and crawlers out. Override with DEMO_TOKEN env var.
const TOKEN = process.env.DEMO_TOKEN || "predicto-demo-2026";
const COOKIE = "predicto_access";

const GATE_PAGE = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Predicto — private demo</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         display: grid; place-items: center; min-height: 100vh; margin: 0;
         background: #f9f9f7; color: #0b0b0b; }
  @media (prefers-color-scheme: dark) { body { background: #0d0d0d; color: #fff; } }
  .card { max-width: 420px; padding: 32px; border: 1px solid rgba(128,128,128,.25);
          border-radius: 12px; text-align: center; }
  h1 { font-size: 18px; margin: 0 0 8px; }
  h1 span { color: #2a78d6; }
  p { font-size: 14px; opacity: .75; line-height: 1.5; }
  input { width: 100%; box-sizing: border-box; padding: 9px 12px; margin: 14px 0 10px;
          border: 1px solid rgba(128,128,128,.35); border-radius: 8px;
          background: transparent; color: inherit; font-size: 14px; }
  button { width: 100%; padding: 9px; border: 0; border-radius: 8px;
           background: #2a78d6; color: #fff; font-size: 14px; cursor: pointer; }
  a { color: #2a78d6; }
</style>
</head>
<body>
  <div class="card">
    <h1>predicto<span>.</span> — private demo</h1>
    <p>This dashboard is token-gated. The access token is published in the
       <a href="https://github.com/alireza1989/predicto#live-demo">README</a>
       of the repository — anyone with repo access is welcome in.</p>
    <form method="GET">
      <input type="password" name="token" placeholder="access token" autofocus>
      <button type="submit">Enter</button>
    </form>
  </div>
</body>
</html>`;

export function middleware(req: NextRequest) {
  const url = req.nextUrl;

  const supplied = url.searchParams.get("token");
  if (supplied === TOKEN) {
    // Set the cookie and drop the token from the visible URL
    const clean = url.clone();
    clean.searchParams.delete("token");
    const res = NextResponse.redirect(clean);
    res.cookies.set(COOKIE, TOKEN, {
      httpOnly: true,
      secure: true,
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 30,
      path: "/",
    });
    return res;
  }

  if (req.cookies.get(COOKIE)?.value === TOKEN) {
    return NextResponse.next();
  }

  return new NextResponse(GATE_PAGE, {
    status: 401,
    headers: { "content-type": "text/html; charset=utf-8" },
  });
}

export const config = {
  // Gate everything except Next.js internals and the favicon
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
