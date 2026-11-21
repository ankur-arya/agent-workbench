import { useEffect, useState, type FormEvent } from "react";
import { del, get, patch, post } from "../api";
import type { Connector } from "../types";

export function Connectors({ onChange }: { onChange: () => void }) {
  const [items, setItems] = useState<Connector[]>([]);
  const [name, setName] = useState("");
  const [serverUrl, setServerUrl] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setItems(await get<Connector[]>("/api/connectors"));
  }

  useEffect(() => {
    void load().catch((err: Error) => setError(err.message));
  }, []);

  async function toggle(connector: Connector) {
    await patch(`/api/connectors/${connector.id}`, { enabled: !connector.enabled });
    await load();
    onChange();
  }

  async function add(event: FormEvent) {
    event.preventDefault();
    await post("/api/connectors", { name, kind: "mcp", server_url: serverUrl, enabled: true, description: "Custom MCP server" });
    setName("");
    setServerUrl("");
    await load();
    onChange();
  }

  async function remove(connector: Connector) {
    await del(`/api/connectors/${connector.id}`);
    await load();
    onChange();
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Connector registry</h1>
          <p className="lede">
            Built-in tools run locally. Legacy OpenAI connectors stay registered and disabled until a token is present and the harness provider is OpenAI.
          </p>
        </div>
      </div>
      {error && <div className="toast">{error}</div>}
      <div className="card" style={{ overflow: "auto" }}>
        <table className="board">
          <thead>
            <tr><th>State</th><th>Kind</th><th>Name</th><th>Id</th><th></th></tr>
          </thead>
          <tbody>
            {items.map((connector) => (
              <tr key={connector.id}>
                <td><button className="btn small" onClick={() => void toggle(connector)}>{connector.enabled ? "Enabled" : "Disabled"}</button></td>
                <td className="mono">{connector.kind}</td>
                <td>{connector.name}<div className="tiny">{connector.description}</div></td>
                <td className="mono">{connector.id}</td>
                <td>{connector.kind !== "builtin" && <button className="btn small" onClick={() => void remove(connector)}>Remove</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <form className="card pad" style={{ marginTop: 14 }} onSubmit={(event) => void add(event)}>
        <h2>Register an MCP server</h2>
        <div className="actions" style={{ marginTop: 12 }}>
          <label className="field">Name<input value={name} onChange={(event) => setName(event.target.value)} required /></label>
          <label className="field">Server URL<input value={serverUrl} onChange={(event) => setServerUrl(event.target.value)} placeholder="https://example.com/mcp" required /></label>
          <button className="btn primary" type="submit">Add connector</button>
        </div>
        <p className="tiny" style={{ marginTop: 10 }}>
          The local provider exposes the connector as a stub tool. The OpenAI provider attaches it as a hosted MCP tool.
        </p>
      </form>
    </>
  );
}
