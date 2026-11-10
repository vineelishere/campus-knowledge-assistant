import type { ReactNode } from "react";

export const metadata = {
  title: "Campus Knowledge Assistant",
  description: "Ask questions about university policies and courses, with cited sources.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
          background: "#f6f7f9",
          color: "#1b1f24",
        }}
      >
        {children}
      </body>
    </html>
  );
}
