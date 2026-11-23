import type { AskStatus } from "../types/api";

export type StatusPresentation = {
  label: string;
  description: string;
  tone: "success" | "warning" | "danger" | "neutral" | "info";
};

const STATUS_MAP: Record<AskStatus, StatusPresentation> = {
  answered: {
    label: "Answered",
    description: "Grounded answer with validated citations from retrieved evidence.",
    tone: "success",
  },
  abstained: {
    label: "Abstained",
    description:
      "The model declined or the answer could not be validated against the evidence.",
    tone: "warning",
  },
  clarification_required: {
    label: "Clarification required",
    description: "Specify whether you mean RoHS or REACH before the assistant can proceed.",
    tone: "info",
  },
  insufficient_evidence: {
    label: "Insufficient evidence",
    description:
      "Retrieval did not return enough relevant regulatory text for this question.",
    tone: "warning",
  },
  out_of_scope: {
    label: "Out of scope",
    description:
      "This assistant answers RoHS and REACH questions from the ingested corpora only.",
    tone: "neutral",
  },
  invalid: {
    label: "Invalid question",
    description: "The question text could not be used (empty, unreadable, or too long).",
    tone: "danger",
  },
};

export function describeAskStatus(status: AskStatus): StatusPresentation {
  return STATUS_MAP[status];
}

export function isSuccessAnswer(status: AskStatus): boolean {
  return status === "answered";
}