import { NavLink } from "react-router-dom";
import { useState } from "react";
import type { Bootstrap } from "../types";

export function Layout({
  boot,
  bleed,
  children,
}: {
  boot: Bootstrap | null;
  bleed?: boolean;
  children: React.ReactNode;
}) {
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || "dark");

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("wb-theme", next);
    setTheme(next);
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            Agent <em>workbench</em>
          </div>
          <div className="brand-sub">Build · Preview · Evaluate</div>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            Workflows <span className="count">{boot?.stats.workflows ?? 0}</span>
          </NavLink>
          <NavLink to="/connectors">
            Connectors <span className="count">{boot?.stats.connectors ?? 0}</span>
          </NavLink>
          <NavLink to="/datasets">
            Datasets <span className="count">{boot?.stats.datasets ?? 0}</span>
          </NavLink>
        </nav>
        <div className="sidebar-foot">
          <div className="pill">
            <span className="dot" />
            Local · Agents SDK
          </div>
          <button className="btn ghost" onClick={toggleTheme}>
            {theme === "dark" ? "Light theme" : "Dark theme"}
          </button>
        </div>
      </aside>
      <main className={bleed ? "main bleed" : "main"}>{children}</main>
    </div>
  );
}
