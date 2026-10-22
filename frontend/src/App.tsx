import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { get } from "./api";
import type { Bootstrap } from "./types";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Studio } from "./pages/Studio";
import { Connectors } from "./pages/Connectors";
import { Datasets } from "./pages/Datasets";

export default function App() {
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [error, setError] = useState<string | null>(null);
  const location = useLocation();
  const bleed = location.pathname.startsWith("/workflows/");

  async function refresh() {
    try {
      setBoot(await get<Bootstrap>("/api/bootstrap"));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the API");
    }
  }

  useEffect(() => {
    void refresh();
  }, [location.pathname]);

  return (
    <Layout boot={boot} bleed={bleed}>
      {error && <div className="toast">{error}</div>}
      <Routes>
        <Route path="/" element={<Home boot={boot} />} />
        <Route path="/workflows/:id" element={<Studio boot={boot} />} />
        <Route path="/connectors" element={<Connectors onChange={refresh} />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
