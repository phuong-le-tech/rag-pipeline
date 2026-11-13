import type { Metadata } from "next";
import "./globals.css";
import { DESCRIPTION, SITE_URL } from "@/lib/site.ts";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: "Measured RAG",
  description: DESCRIPTION,
  openGraph: { title: "Measured RAG", description: DESCRIPTION, type: "website", url: "/" },
  twitter: { card: "summary_large_image", title: "Measured RAG", description: DESCRIPTION },
};

// Theme before first paint: ?theme= wins, then saved choice, default dark.
const themeScript = `(function(){try{var q=new URLSearchParams(location.search).get("theme");var t=(q==="light"||q==="dark")?q:localStorage.getItem("gb-theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t);}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      {/* Browser extensions inject attributes on <body> before hydration. */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
