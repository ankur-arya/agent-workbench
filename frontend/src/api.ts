const JSON_HEADERS = { "Content-Type": "application/json" };

export function errorDetailFromBody(raw: string, fallback: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return fallback;
  try {
    const body = JSON.parse(trimmed) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (body.detail != null) return JSON.stringify(body.detail);
    return JSON.stringify(body);
  } catch {
    return raw;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  const fallback = response.statusText || `Request failed (${response.status})`;
  const raw = await response.text();
  return errorDetailFromBody(raw, fallback);
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    const detail = await readErrorDetail(response);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const get = <T>(path: string) => api<T>(path);

export const post = <T>(path: string, body?: unknown) =>
  api<T>(path, {
    method: "POST",
    headers: JSON_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const put = <T>(path: string, body?: unknown) =>
  api<T>(path, {
    method: "PUT",
    headers: JSON_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const patch = <T>(path: string, body?: unknown) =>
  api<T>(path, {
    method: "PATCH",
    headers: JSON_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const del = <T>(path: string) => api<T>(path, { method: "DELETE" });
