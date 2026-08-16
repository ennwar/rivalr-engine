import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "rivalr",
  description: "Mini-league FPL intelligence",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0c0f14",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <nav className="topnav">
          <a href="/">brief</a>
          <a href="/fixtures">fixtures</a>
          <a href="/planner">planner</a>
          <a href="/accuracy">accuracy</a>
        </nav>
        {children}
      </body>
    </html>
  );
}
