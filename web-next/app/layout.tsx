import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Predicto — Meta Data-Scientist",
  description:
    "Autonomous multi-agent NBA prediction platform: experiments, market edges, CLV.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <nav className="topnav">
            <Link href="/" className="brand">
              predicto<span>.</span>
            </Link>
            <Link href="/" className="nav">
              Overview
            </Link>
            <Link href="/experiments" className="nav">
              Experiments
            </Link>
            <Link href="/predictions" className="nav">
              Predictions
            </Link>
            <Link href="/performance" className="nav">
              Performance
            </Link>
          </nav>
          {children}
        </div>
      </body>
    </html>
  );
}
