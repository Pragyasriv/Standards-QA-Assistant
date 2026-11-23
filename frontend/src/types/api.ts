export type AskStatus =
  | "invalid"
  | "out_of_scope"
  | "clarification_required"
  | "insufficient_evidence"
  | "answered"
  | "abstained";

export interface ResolvedCitation {
  chunk_id: string;
  document_id: string;
  regulation: string;
  regulation_number: string | null;
  article: string | null;
  paragraph: string | null;
  subparagraph: string | null;
  annex: string | null;
  page_start: number | null;
  page_end: number | null;
  source_filename: string | null;
}

export interface GenerationMetadata {
  evidence_chunk_count: number;
  validated_citation_count: number;
  abstained: boolean;
  latency_ms: number;
  temperature: number;
}

export interface AskRequest {
  question: string;
}

export interface AskResponse {
  question: string;
  status: AskStatus;
  answer: string | null;
  reason: string | null;
  clarification_question: string | null;
  citations: ResolvedCitation[];
  abstention_reason: string | null;
  generation_metadata: GenerationMetadata | null;
}

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ReadinessResponse {
  status: "ready" | "not_ready";
  qdrant_available: boolean;
  bm25_index_available: boolean;
}

export interface ErrorResponse {
  error: string;
  message: string;
  details?: Array<Record<string, unknown>> | null;
}