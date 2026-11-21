import { useEffect, useState, type FormEvent } from "react";
import { get, post } from "../api";
import type { Workflow } from "../types";

type Dataset = { id: string; name: string; workflow_id: string | null; tasks: Array<{ input: string }> };
type EvalRun = { id: string; workflow_id: string; summary: { passed?: number; failed?: number; total?: number }; created_at: string };

export function Datasets() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [name, setName] = useState("Sample tasks");
  const [workflowId, setWorkflowId] = useState("");
  const [tasks, setTasks] = useState("What is 2 + 2?\nexpect_contains: 4");
  const [error, setError] = useState<string | null>(null);

  async function load() {
    const [workflowRows, datasetRows, evalRows] = await Promise.all([
      get<Workflow[]>("/api/workflows"),
      get<Dataset[]>("/api/datasets"),
      get<EvalRun[]>("/api/evals"),
    ]);
    setWorkflows(workflowRows);
    setDatasets(datasetRows);
    setRuns(evalRows);
    if (!workflowId && workflowRows[0]) setWorkflowId(workflowRows[0].id);
  }

  useEffect(() => {
    void load().catch((err: Error) => setError(err.message));
  }, []);

  async function create(event: FormEvent) {
    event.preventDefault();
    const parsed = tasks.split("\n").map((line) => line.trim()).filter(Boolean);
    const input = parsed[0] || "";
    const expect = parsed.find((line) => line.startsWith("expect_contains:"))?.split(":").slice(1).join(":").trim();
    await post("/api/datasets", {
      name,
      workflow_id: workflowId || null,
      tasks: [{ input, expect_contains: expect || null }],
    });
    await load();
  }

  async function grade(dataset: Dataset) {
    if (!dataset.workflow_id) return;
    await post(`/api/workflows/${dataset.workflow_id}/eval`, { dataset_id: dataset.id });
    await load();
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Datasets and evals</h1>
          <p className="lede">
            Sample tasks are the local stand-in for eval datasets. Graders check strings, tools, handoffs, guardrails, similarity, and word count.
          </p>
        </div>
      </div>
      {error && <div className="toast">{error}</div>}
      <form className="card pad" onSubmit={(event) => void create(event)}>
        <div className="actions">
          <label className="field">Name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label className="field">Workflow
            <select value={workflowId} onChange={(event) => setWorkflowId(event.target.value)}>
              {workflows.map((workflow) => <option key={workflow.id} value={workflow.id}>{workflow.name}</option>)}
            </select>
          </label>
        </div>
        <label className="field" style={{ marginTop: 12 }}>Tasks
          <textarea value={tasks} onChange={(event) => setTasks(event.target.value)} />
        </label>
        <button className="btn primary" style={{ marginTop: 12 }} type="submit">Save dataset</button>
      </form>
      <div className="grid-cards" style={{ marginTop: 14 }}>
        {datasets.map((dataset) => (
          <article key={dataset.id} className="card exp-card">
            <h3>{dataset.name}</h3>
            <p className="tiny">{dataset.tasks.length} tasks</p>
            <button className="btn" onClick={() => void grade(dataset)}>Grade workflow</button>
          </article>
        ))}
      </div>
      <div className="card" style={{ marginTop: 14 }}>
        <table className="board">
          <thead><tr><th>When</th><th>Workflow</th><th>Passed</th><th>Failed</th></tr></thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}>
                <td className="tiny">{run.created_at}</td>
                <td className="mono">{run.workflow_id}</td>
                <td>{run.summary.passed ?? 0}</td>
                <td>{run.summary.failed ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
