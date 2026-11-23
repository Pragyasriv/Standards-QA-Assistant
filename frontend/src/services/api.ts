import type {
  AskRequest,
  AskResponse,
  ErrorResponse,
  HealthResponse,
  ReadinessResponse,
} from "../types/api";

export class ApiError extends Error {
  readonly status: number;
  readonly errorCode: string | undefined;
  readonly details: ErrorResponse["details"];

  constructor(
    status: number,
    message: string,
    errorCode?: string,
    details?: ErrorResponse["details"],
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errorCode = errorCode;
    this.details = details;
  }
}

async function parseError(response: Response): Promise<ApiError> {
  let body: ErrorResponse | null = null;
  try {
    body = (await response.json()) as ErrorResponse;
  } catch {
    body = null;
  }
  const message =
    body?.message ??
    `Request failed with status ${response.status} (${response.statusText}).`;
  return new ApiError(response.status, message, body?.error, body?.details);
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError(
      0,
      "Unable to reach the API. Ensure the FastAPI server is running on port 8000.",
    );
  }
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export function fetchReady(): Promise<ReadinessResponse> {
  return requestJson<ReadinessResponse>("/ready");
}

export function postAsk(body: AskRequest): Promise<AskResponse> {
  return requestJson<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}