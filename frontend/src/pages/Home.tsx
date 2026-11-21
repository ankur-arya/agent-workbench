import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { get, post } from "../api";
import type { Bootstrap, Workflow } from "../types";

export function Home({ boot }: { boot: Bootstrap | null }) {
  const [items, setItems] = useState<Workflow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    void get<Workflow[]>("/api/workflows")
      .then(setItems)
      .catch((err: Error) => setError(err.message));
  }, []);

  async function create(template: string) {
    setBusy(template);
    setError(null);
    try {
      const created = await post<Workflow>("/api/workflows", { template });
      navigate(`/workflows/${created.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the workflow");
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Workflows</h1>
          <p className="lede">
            Start from a template, then configure tools, connectors, guardrails, and handoffs.
            Publish a version when the preview looks right.
          </p>
        </div>
      </div>
      {error && <div className="toast">{error}</div>}
      <div className="grid-cards" style={{ marginBottom: 22 }}>
        {(boot?.templates ?? []).map((template) => (
          <button key={template.id} className="choice" disabled={busy === template.id} onClick={() => void create(template.id)}>
            <h3>{template.name}</h3>
            <p>{template.description}</p>
          </button>
        ))}
      </div>
      {items.length > 0 && (
        <div className="grid-cards">
          {items.map((item) => (
            <Link key={item.id} to={`/workflows/${item.id}`} className="card exp-card">
              <div className="top">
                <h3>{item.name}</h3>
                <span className={`status ${item.status}`}>{item.status}</span>
              </div>
              <p className="tiny">{item.description}</p>
              <div className="meta">
                <span>{item.template || "custom"}</span>
                <span>{item.document.harness.kind}</span>
                <span>v{item.current_version}</span>
              </div>
            </Link>
          ))}
        </div>
      )}
      {boot && (
        <p className="tiny" style={{ marginTop: 18 }}>
          Store {boot.home} · {boot.stats.traces} traces · local provider needs no API key
        </p>
      )}
    </>
  );
}
