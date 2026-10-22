export type Harness = {
  kind: string;
  max_turns: number;
  tracing: boolean;
  provider: "local" | "openai";
  model: string | null;
  temperature: number | null;
  tool_choice: "auto" | "required" | "none" | null;
  parallel_tool_calls: boolean | null;
  tool_use_behavior: "run_llm_again" | "stop_on_first_tool";
  session_backend: "none" | "memory" | "sqlite";
  session_id: string | null;
  approval: "auto" | "manual" | "reject";
  eval_mode: "sync" | "async";
};

export type NodeModel = {
  id: string;
  type: string;
  x: number;
  y: number;
  config: Record<string, unknown>;
};

export type EdgeModel = {
  id: string;
  source: string;
  target: string;
  source_handle: string;
  target_handle: string;
};

export type EvalTask = {
  input: string;
  expect_contains?: string | null;
  expect_exact?: string | null;
  expect_tool?: string | null;
  expect_handoff?: string | null;
  expect_guardrail?: boolean | null;
  expect_status?: string | null;
  similarity_reference?: string | null;
  similarity_threshold?: number;
  max_words?: number | null;
};

export type WorkflowDocument = {
  name: string;
  description: string;
  template: string | null;
  harness: Harness;
  nodes: NodeModel[];
  edges: EdgeModel[];
  tasks: EvalTask[];
};

export type Workflow = {
  id: string;
  name: string;
  description: string;
  template: string | null;
  status: string;
  document: WorkflowDocument;
  current_version: number;
  created_at: string;
  updated_at: string;
};

export type TemplateInfo = { id: string; name: string; description: string };
export type HarnessInfo = { kind: string; summary: string };
export type NodeInfo = { type: string; group: string; summary: string };

export type Connector = {
  id: string;
  name: string;
  kind: string;
  description: string;
  enabled: boolean;
  server_url: string | null;
  requires_approval: boolean;
};

export type Bootstrap = {
  version: string;
  home: string;
  stats: Record<string, number>;
  templates: TemplateInfo[];
  harnesses: HarnessInfo[];
  nodes: NodeInfo[];
  tools: Connector[];
};

export type TraceEvent = {
  type: string;
  name?: string;
  detail?: string;
  node_id?: string;
};

export type Trace = {
  id: string;
  workflow_id: string;
  thread_id: string | null;
  status: string;
  input: string;
  output: string;
  last_agent: string | null;
  events: TraceEvent[];
  pending: Record<string, unknown> | null;
  harness: string;
  created_at: string;
};

export type Version = {
  id: string;
  workflow_id: string;
  version: number;
  note: string;
  created_at: string;
};

export type ChatMessage = {
  id: string;
  role: string;
  content: string;
  trace_id: string | null;
};
