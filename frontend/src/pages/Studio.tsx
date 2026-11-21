import { useEffect, useMemo, useState, type PointerEvent as ReactPointerEvent } from "react";
import { useParams } from "react-router-dom";
import { get, post, put } from "../api";
import type { Bootstrap, ChatMessage, EdgeModel, NodeModel, Trace, Version, Workflow, WorkflowDocument } from "../types";

const HANDLES: Record<string, string[]> = {
  start: ["out"],
  agent: ["out", "handoff", "tool"],
  note: ["out"],
  file_search: ["out"],
  guardrails: ["pass", "fail"],
  mcp: ["out"],
  if_else: ["true", "false"],
  while: ["body", "done"],
  human_approval: ["approved", "rejected"],
  transform: ["out"],
  set_state: ["out"],
};

function rid(prefix: string) {
  return `${prefix}_${Math.random().toString(16).slice(2, 8)}`;
}

export function Studio({ boot }: { boot: Bootstrap | null }) {
  const { id = "" } = useParams();
  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [doc, setDoc] = useState<WorkflowDocument | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [link, setLink] = useState<{ node: string; handle: string } | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [traces, setTraces] = useState<Trace[]>([]);
  const [tab, setTab] = useState<"preview" | "traces" | "evals" | "code">("preview");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [live, setLive] = useState("");
  const [pendingTrace, setPendingTrace] = useState<string | null>(null);
  const [report, setReport] = useState<string>("");
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void (async () => {
      const item = await get<Workflow>(`/api/workflows/${id}`);
      setWorkflow(item);
      setDoc(structuredClone(item.document));
      setVersions(await get<Version[]>(`/api/workflows/${id}/versions`));
      setTraces(await get<Trace[]>(`/api/workflows/${id}/traces`));
    })().catch((err: Error) => setError(err.message));
  }, [id]);

  const selectedNode = doc?.nodes.find((node) => node.id === selected) ?? null;

  async function persist(next: WorkflowDocument) {
    const saved = await put<Workflow>(`/api/workflows/${id}`, { document: next });
    setWorkflow(saved);
    return saved;
  }

  function updateDoc(mutator: (current: WorkflowDocument) => WorkflowDocument) {
    setDoc((current) => (current ? mutator(current) : current));
  }

  async function save() {
    if (!doc) return;
    setBusy(true);
    try {
      await persist(doc);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setBusy(false);
    }
  }

  function addNode(type: string) {
    updateDoc((current) => {
      const node: NodeModel = {
        id: rid(type),
        type,
        x: 80 + current.nodes.length * 28,
        y: 80 + (current.nodes.length % 4) * 36,
        config: defaultConfig(type),
      };
      return { ...current, nodes: [...current.nodes, node] };
    });
  }

  function move(nodeId: string, x: number, y: number) {
    updateDoc((current) => ({
      ...current,
      nodes: current.nodes.map((node) => (node.id === nodeId ? { ...node, x, y } : node)),
    }));
  }

  function connect(target: string) {
    if (!link || !doc || link.node === target) return;
    const edge: EdgeModel = {
      id: rid("e"),
      source: link.node,
      target,
      source_handle: link.handle,
      target_handle: "in",
    };
    updateDoc((current) => ({ ...current, edges: [...current.edges, edge] }));
    setLink(null);
  }

  function removeSelected() {
    if (!selected) return;
    updateDoc((current) => ({
      ...current,
      nodes: current.nodes.filter((node) => node.id !== selected),
      edges: current.edges.filter((edge) => edge.source !== selected && edge.target !== selected),
    }));
    setSelected(null);
  }

  async function publish() {
    if (!doc) return;
    setBusy(true);
    try {
      await persist(doc);
      await post(`/api/workflows/${id}/publish`, { note: "Published from the canvas" });
      const fresh = await get<Workflow>(`/api/workflows/${id}`);
      setWorkflow(fresh);
      setVersions(await get(`/api/workflows/${id}/versions`));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Publish failed");
    } finally {
      setBusy(false);
    }
  }

  async function restore(versionId: string) {
    const restored = await post<Workflow>(`/api/workflows/${id}/restore`, { version_id: versionId });
    setWorkflow(restored);
    setDoc(structuredClone(restored.document));
  }

  async function ensureThread() {
    if (threadId) return threadId;
    const thread = await post<{ id: string }>("/api/threads", { workflow_id: id, title: "Preview" });
    setThreadId(thread.id);
    return thread.id;
  }

  async function send() {
    if (!doc || !draft.trim()) return;
    const text = draft.trim();
    setDraft("");
    setLive("");
    setBusy(true);
    setError(null);
    try {
      await persist(doc);
      const currentThread = await ensureThread();
      setMessages((items) => [...items, { id: rid("msg"), role: "user", content: text, trace_id: null }]);
      if (doc.harness.kind === "streaming") {
        await streamMessage(currentThread, text);
      } else {
        const response = await post<{
          trace: Trace;
          outcome: { output: string; status: string };
          thread: { messages: ChatMessage[] };
        }>(`/api/threads/${currentThread}/messages`, { text });
        setMessages(response.thread.messages);
        setPendingTrace(response.outcome.status === "needs_approval" ? response.trace.id : null);
        setTraces(await get(`/api/workflows/${id}/traces`));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run failed");
    } finally {
      setBusy(false);
    }
  }

  async function streamMessage(currentThread: string, text: string) {
    const response = await fetch(`/api/threads/${currentThread}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, stream: true }),
    });
    if (!response.ok || !response.body) throw new Error("Streaming run failed");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let acc = "";
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const line = part.split("\n").find((item) => item.startsWith("data:"));
        if (!line) continue;
        const payload = JSON.parse(line.slice(5).trim()) as {
          type: string;
          text?: string;
          error?: string;
          outcome?: { output: string; status: string };
          trace?: Trace;
        };
        if (payload.type === "delta" && payload.text) {
          acc += payload.text;
          setLive(acc);
        }
        if (payload.type === "error") throw new Error(payload.error || "stream error");
        if (payload.type === "done" && payload.outcome && payload.trace) {
          setMessages((items) => [
            ...items,
            { id: rid("msg"), role: "assistant", content: payload.outcome?.output || acc, trace_id: payload.trace?.id || null },
          ]);
          setLive("");
          setPendingTrace(payload.outcome.status === "needs_approval" ? payload.trace.id : null);
        }
      }
    }
    setTraces(await get(`/api/workflows/${id}/traces`));
  }

  async function decide(decision: "approve" | "reject") {
    if (!pendingTrace) return;
    setBusy(true);
    try {
      const response = await post<{ outcome: { output: string; status: string }; trace: Trace }>(
        `/api/traces/${pendingTrace}/resume`,
        { decision },
      );
      setMessages((items) => [
        ...items,
        { id: rid("msg"), role: "assistant", content: response.outcome.output, trace_id: response.trace.id },
      ]);
      setPendingTrace(response.outcome.status === "needs_approval" ? response.trace.id : null);
      setTraces(await get(`/api/workflows/${id}/traces`));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Approval failed");
    } finally {
      setBusy(false);
    }
  }

  async function runEval() {
    if (!doc) return;
    setBusy(true);
    setTab("evals");
    try {
      await persist(doc);
      const response = await post<{ outcome: { output: string; report: { results: Array<{ input: string; passed: boolean; output: string }> } } }>(
        `/api/workflows/${id}/eval`,
        {},
      );
      const lines = (response.outcome.report?.results ?? []).map(
        (row) => `${row.passed ? "PASS" : "FAIL"}  ${row.input}\n    ${row.output}`,
      );
      setReport(`${response.outcome.output}\n\n${lines.join("\n")}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Eval failed");
    } finally {
      setBusy(false);
    }
  }

  async function showCode() {
    setTab("code");
    if (doc) await persist(doc);
    const response = await get<{ python: string }>(`/api/workflows/${id}/code`);
    setCode(response.python);
  }

  const size = useMemo(() => {
    const nodes = doc?.nodes ?? [];
    return {
      width: Math.max(980, ...nodes.map((node) => node.x + 240)),
      height: Math.max(520, ...nodes.map((node) => node.y + 160)),
    };
  }, [doc]);

  if (!doc || !workflow) return <p className="tiny">Loading workflow…</p>;

  return (
    <div className="studio">
      {error && <div className="toast">{error}</div>}
      <div className="studio-bar">
        <div>
          <h1>{doc.name}</h1>
          <p className="tiny">{workflow.status} · v{workflow.current_version} · {doc.harness.kind} · {doc.harness.provider}</p>
        </div>
        <div className="actions">
          <button className="btn" disabled={busy} onClick={() => void save()}>Save draft</button>
          <button className="btn gold" disabled={busy} onClick={() => void publish()}>Publish</button>
          <button className="btn" disabled={busy} onClick={() => void runEval()}>Run eval</button>
          <select className="btn" value="" onChange={(event) => event.target.value && void restore(event.target.value)}>
            <option value="">Versions</option>
            {versions.map((version) => (
              <option key={version.id} value={version.id}>v{version.version} {version.note}</option>
            ))}
          </select>
        </div>
      </div>

      <aside className="palette">
        <h2>Nodes</h2>
        {["Core", "Tools", "Logic", "Data"].map((group) => (
          <div key={group}>
            <div className="group-label">{group}</div>
            {(boot?.nodes ?? []).filter((node) => node.group === group).map((node) => (
              <button key={node.type} className="choice" onClick={() => addNode(node.type)} title={node.summary}>
                <h3>{node.type.replaceAll("_", " ")}</h3>
                <p>{node.summary}</p>
              </button>
            ))}
          </div>
        ))}
      </aside>

      <div className="canvas-wrap">
        <div className="canvas" style={{ width: size.width, height: size.height }} onClick={() => setSelected(null)}>
          <svg>
            {doc.edges.map((edge) => {
              const source = doc.nodes.find((node) => node.id === edge.source);
              const target = doc.nodes.find((node) => node.id === edge.target);
              if (!source || !target) return null;
              const x1 = source.x + 196;
              const y1 = source.y + 36;
              const x2 = target.x;
              const y2 = target.y + 36;
              return (
                <g key={edge.id}>
                  <path d={`M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`} stroke="currentColor" strokeOpacity="0.45" fill="none" />
                  <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} fill="currentColor" fontSize="10">{edge.source_handle}</text>
                </g>
              );
            })}
          </svg>
          {doc.nodes.map((node) => (
            <article
              key={node.id}
              className={`flow-node ${selected === node.id ? "selected" : ""}`}
              style={{ left: node.x, top: node.y }}
              onClick={(event) => {
                event.stopPropagation();
                if (link) connect(node.id);
                else setSelected(node.id);
              }}
              onPointerDown={(event) => startDrag(event, node, move)}
            >
              <span className="kind">{node.type}</span>
              <strong>{String(node.config.name || node.config.key || node.id)}</strong>
              <div className="ports">
                {(HANDLES[node.type] ?? ["out"]).map((handle) => (
                  <button
                    key={handle}
                    className={`port ${link?.node === node.id && link.handle === handle ? "on" : ""}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      setLink({ node: node.id, handle });
                    }}
                  >
                    {handle}
                  </button>
                ))}
              </div>
            </article>
          ))}
        </div>
      </div>

      <aside className="inspector">
        <h2>{selectedNode ? selectedNode.type : "Harness"}</h2>
        <div className="stack">
          {!selectedNode && (
            <>
              <label className="field">Name
                <input value={doc.name} onChange={(event) => updateDoc((current) => ({ ...current, name: event.target.value }))} />
              </label>
              <label className="field">Description
                <textarea value={doc.description} onChange={(event) => updateDoc((current) => ({ ...current, description: event.target.value }))} />
              </label>
              <label className="field">Harness
                <select value={doc.harness.kind} onChange={(event) => updateDoc((current) => ({ ...current, harness: { ...current.harness, kind: event.target.value } }))}>
                  {(boot?.harnesses ?? []).map((item) => <option key={item.kind} value={item.kind}>{item.kind}</option>)}
                </select>
              </label>
              <p className="tiny">{boot?.harnesses.find((item) => item.kind === doc.harness.kind)?.summary}</p>
              <label className="field">Provider
                <select value={doc.harness.provider} onChange={(event) => updateDoc((current) => ({ ...current, harness: { ...current.harness, provider: event.target.value as "local" | "openai" } }))}>
                  <option value="local">local</option>
                  <option value="openai">openai</option>
                </select>
              </label>
              <label className="field">Max turns
                <input type="number" value={doc.harness.max_turns} onChange={(event) => updateDoc((current) => ({ ...current, harness: { ...current.harness, max_turns: Number(event.target.value) } }))} />
              </label>
              <label className="field">Tool choice
                <select value={doc.harness.tool_choice ?? "auto"} onChange={(event) => updateDoc((current) => ({ ...current, harness: { ...current.harness, tool_choice: event.target.value as "auto" | "required" | "none" } }))}>
                  <option value="auto">auto</option>
                  <option value="required">required</option>
                  <option value="none">none</option>
                </select>
              </label>
              <label className="field">Tool use
                <select value={doc.harness.tool_use_behavior} onChange={(event) => updateDoc((current) => ({ ...current, harness: { ...current.harness, tool_use_behavior: event.target.value as "run_llm_again" | "stop_on_first_tool" } }))}>
                  <option value="run_llm_again">run_llm_again</option>
                  <option value="stop_on_first_tool">stop_on_first_tool</option>
                </select>
              </label>
              <label className="field">Session
                <select value={doc.harness.session_backend} onChange={(event) => updateDoc((current) => ({ ...current, harness: { ...current.harness, session_backend: event.target.value as "none" | "memory" | "sqlite" } }))}>
                  <option value="none">none</option>
                  <option value="memory">memory</option>
                  <option value="sqlite">sqlite</option>
                </select>
              </label>
              <label className="field">Approval
                <select value={doc.harness.approval} onChange={(event) => updateDoc((current) => ({ ...current, harness: { ...current.harness, approval: event.target.value as "auto" | "manual" | "reject" } }))}>
                  <option value="manual">manual</option>
                  <option value="auto">auto</option>
                  <option value="reject">reject</option>
                </select>
              </label>
              <label className="check-row">
                <input type="checkbox" checked={doc.harness.tracing} onChange={(event) => updateDoc((current) => ({ ...current, harness: { ...current.harness, tracing: event.target.checked } }))} />
                Tracing export when an OpenAI key is set
              </label>
            </>
          )}
          {selectedNode && (
            <NodeInspector
              node={selectedNode}
              tools={boot?.tools ?? []}
              onChange={(config) => updateDoc((current) => ({
                ...current,
                nodes: current.nodes.map((node) => node.id === selectedNode.id ? { ...node, config } : node),
              }))}
              onDelete={removeSelected}
            />
          )}
        </div>
      </aside>

      <section className="dock">
        <div className="tabs">
          <button className={`tab ${tab === "preview" ? "on" : ""}`} onClick={() => setTab("preview")}>Preview</button>
          <button className={`tab ${tab === "traces" ? "on" : ""}`} onClick={() => setTab("traces")}>Traces</button>
          <button className={`tab ${tab === "evals" ? "on" : ""}`} onClick={() => setTab("evals")}>Evals</button>
          <button className={`tab ${tab === "code" ? "on" : ""}`} onClick={() => void showCode()}>Code</button>
        </div>
        <div className="dock-body">
          {tab === "preview" && (
            <div className="chat">
              <div className="transcript">
                {messages.map((message) => (
                  <div key={message.id} className={`bubble ${message.role}`}>{message.content}</div>
                ))}
                {live && <div className="bubble assistant">{live}</div>}
                {messages.length === 0 && <p className="tiny">Run the draft in this chat. Streaming harnesses show tokens as they arrive.</p>}
              </div>
              {pendingTrace && (
                <div className="actions">
                  <button className="btn gold" disabled={busy} onClick={() => void decide("approve")}>Approve</button>
                  <button className="btn" disabled={busy} onClick={() => void decide("reject")}>Reject</button>
                </div>
              )}
              <form className="composer" onSubmit={(event) => { event.preventDefault(); void send(); }}>
                <input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask the agent" />
                <button className="btn primary" disabled={busy} type="submit">{busy ? "Running…" : "Send"}</button>
              </form>
            </div>
          )}
          {tab === "traces" && (
            <div>
              {traces.length === 0 && <p className="tiny">No traces yet. Send a preview message.</p>}
              {traces.map((trace) => (
                <details key={trace.id} open={trace.id === traces[0]?.id}>
                  <summary className="tiny">{trace.status} · {trace.harness} · {trace.input}</summary>
                  <p style={{ margin: "8px 0" }}>{trace.output}</p>
                  {trace.events.map((event, index) => (
                    <div className="event" key={`${trace.id}-${index}`}>
                      <b>{event.type}</b>
                      <span className="mono">{event.name}</span>
                      <span>{event.detail}</span>
                    </div>
                  ))}
                </details>
              ))}
            </div>
          )}
          {tab === "evals" && <pre className="code">{report || "Run eval to grade this workflow's sample tasks."}</pre>}
          {tab === "code" && <pre className="code">{code || "Open this tab to export Agents SDK Python."}</pre>}
        </div>
      </section>
    </div>
  );
}

function NodeInspector({
  node,
  tools,
  onChange,
  onDelete,
}: {
  node: NodeModel;
  tools: Bootstrap["tools"];
  onChange: (config: Record<string, unknown>) => void;
  onDelete: () => void;
}) {
  const config = node.config;
  function set(key: string, value: unknown) {
    onChange({ ...config, [key]: value });
  }
  const builtins = tools.filter((tool) => tool.kind === "builtin");
  const selectedTools = new Set((config.tools as string[] | undefined) ?? []);
  return (
    <>
      {node.type === "agent" && (
        <>
          <label className="field">Name<input value={String(config.name ?? "")} onChange={(event) => set("name", event.target.value)} /></label>
          <label className="field">Instructions<textarea value={String(config.instructions ?? "")} onChange={(event) => set("instructions", event.target.value)} /></label>
          <label className="field">Handoff description<input value={String(config.handoff_description ?? "")} onChange={(event) => set("handoff_description", event.target.value)} /></label>
          <div className="group-label">Tools</div>
          {builtins.map((tool) => (
            <label key={tool.id} className="check-row">
              <input
                type="checkbox"
                checked={selectedTools.has(tool.id)}
                onChange={(event) => {
                  const next = new Set(selectedTools);
                  if (event.target.checked) next.add(tool.id);
                  else next.delete(tool.id);
                  set("tools", [...next]);
                }}
              />
              {tool.name}
            </label>
          ))}
          <label className="field">Input guardrails
            <input value={((config.input_guardrails as string[] | undefined) ?? []).join(", ")} onChange={(event) => set("input_guardrails", splitList(event.target.value))} />
          </label>
          <label className="field">Output guardrails
            <input value={((config.output_guardrails as string[] | undefined) ?? []).join(", ")} onChange={(event) => set("output_guardrails", splitList(event.target.value))} />
          </label>
        </>
      )}
      {node.type === "note" && (
        <label className="field">Note<textarea value={String(config.text ?? "")} onChange={(event) => set("text", event.target.value)} /></label>
      )}
      {node.type === "guardrails" && (
        <>
          <label className="field">Checks
            <input value={((config.checks as string[] | undefined) ?? []).join(", ")} onChange={(event) => set("checks", splitList(event.target.value))} />
          </label>
          <label className="field">Action
            <select value={String(config.action ?? "flag")} onChange={(event) => set("action", event.target.value)}>
              <option value="flag">flag</option>
              <option value="mask">mask</option>
            </select>
          </label>
        </>
      )}
      {node.type === "if_else" && (
        <label className="field">Expression
          <input value={String(config.expression ?? "")} onChange={(event) => set("expression", event.target.value)} />
        </label>
      )}
      {node.type === "while" && (
        <>
          <label className="field">Expression<input value={String(config.expression ?? "")} onChange={(event) => set("expression", event.target.value)} /></label>
          <label className="field">Max iterations<input type="number" value={Number(config.max_iterations ?? 3)} onChange={(event) => set("max_iterations", Number(event.target.value))} /></label>
        </>
      )}
      {node.type === "human_approval" && (
        <label className="field">Prompt<input value={String(config.prompt ?? "")} onChange={(event) => set("prompt", event.target.value)} /></label>
      )}
      {node.type === "set_state" && (
        <>
          <label className="field">Key<input value={String(config.key ?? "")} onChange={(event) => set("key", event.target.value)} /></label>
          <label className="field">Source
            <select value={String(config.source ?? "output")} onChange={(event) => set("source", event.target.value)}>
              <option value="output">output</option>
              <option value="input">input</option>
            </select>
          </label>
        </>
      )}
      {node.type === "transform" && (
        <>
          <label className="field">Mode
            <select value={String(config.mode ?? "wrap")} onChange={(event) => set("mode", event.target.value)}>
              <option value="wrap">wrap</option>
              <option value="set">set</option>
              <option value="append">append</option>
              <option value="extract_json">extract_json</option>
            </select>
          </label>
          <label className="field">Prefix or value<input value={String(config.prefix ?? config.value ?? "")} onChange={(event) => set(config.mode === "set" || config.mode === "append" ? "value" : "prefix", event.target.value)} /></label>
        </>
      )}
      {node.type === "mcp" && (
        <label className="field">Connector id<input value={String(config.connector_id ?? "")} onChange={(event) => set("connector_id", event.target.value)} /></label>
      )}
      <button className="btn" onClick={onDelete}>Remove node</button>
    </>
  );
}

function splitList(value: string) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function defaultConfig(type: string): Record<string, unknown> {
  if (type === "agent") return { name: "New agent", instructions: "You are a helpful agent.", tools: [] };
  if (type === "note") return { text: "Note" };
  if (type === "guardrails") return { checks: ["pii", "jailbreak"], action: "flag", source: "input" };
  if (type === "if_else") return { expression: 'contains(output, "yes")' };
  if (type === "while") return { expression: "not_empty(output)", max_iterations: 2 };
  if (type === "human_approval") return { prompt: "Approve this step?" };
  if (type === "set_state") return { key: "query", source: "output" };
  if (type === "transform") return { mode: "wrap", prefix: "Result" };
  if (type === "file_search") return { name: "File search" };
  if (type === "mcp") return { connector_id: "" };
  return {};
}

function startDrag(event: ReactPointerEvent, node: NodeModel, move: (id: string, x: number, y: number) => void) {
  const target = event.target as HTMLElement;
  if (target.closest("button")) return;
  event.preventDefault();
  const originX = event.clientX;
  const originY = event.clientY;
  const startX = node.x;
  const startY = node.y;
  function onMove(ev: globalThis.PointerEvent) {
    move(node.id, Math.max(12, startX + ev.clientX - originX), Math.max(12, startY + ev.clientY - originY));
  }
  function onUp() {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  }
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
}
