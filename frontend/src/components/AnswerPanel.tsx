import { AlertCircle, FileText, Info, XCircle } from "lucide-react";
import type { AskResponse } from "../types/api";
import { describeAskStatus, isSuccessAnswer } from "../utils/status";
import { AnswerContent } from "./AnswerContent";
import { CitationCard } from "./CitationCard";
import { LoadingPanel } from "./LoadingPanel";

type Props = {
  response: AskResponse | null;
  error: string | null;
  loading: boolean;
};

const TONE_STYLES = {
  success: "border-emerald-200/80 bg-emerald-50/90 text-emerald-950",
  warning: "border-amber-200/80 bg-amber-50/95 text-amber-950",
  danger: "border-rose-200/80 bg-rose-50/95 text-rose-950",
  neutral: "border-slate-200 bg-slate-50 text-slate-800",
  info: "border-sky-200/80 bg-sky-50/95 text-sky-950",
} as const;

const STATUS_ICONS = {
  success: FileText,
  warning: Info,
  danger: XCircle,
  neutral: Info,
  info: Info,
} as const;

export function AnswerPanel({ response, error, loading }: Props) {
  if (loading) {
    return <LoadingPanel />;
  }

  if (error) {
    return (
      <section
        className="panel-elevated border-rose-200/80 bg-rose-50/95 p-5 sm:p-6"
        role="alert"
      >
        <div className="flex gap-3">
          <AlertCircle className="h-5 w-5 shrink-0 text-rose-700" />
          <div>
            <h3 className="text-sm font-semibold text-rose-950">Request failed</h3>
            <p className="mt-2 text-sm leading-relaxed text-rose-900">{error}</p>
          </div>
        </div>
      </section>
    );
  }

  if (!response) return null;

  const statusInfo = describeAskStatus(response.status);
  const success = isSuccessAnswer(response.status);
  const StatusIcon = STATUS_ICONS[statusInfo.tone];

  return (
    <div className="space-y-5">
      <section
        className={`rounded-xl border px-4 py-4 sm:px-5 ${TONE_STYLES[statusInfo.tone]}`}
        aria-label={`Status: ${statusInfo.label}`}
      >
        <div className="flex items-start gap-3">
          <StatusIcon className="mt-0.5 h-5 w-5 shrink-0 opacity-90" />
          <div>
            <p className="text-label text-current opacity-80">
              {statusInfo.label}
            </p>
            <p className="mt-1 text-sm leading-relaxed">{statusInfo.description}</p>
          </div>
        </div>
      </section>

      {success && response.answer && (
        <section className="panel-elevated overflow-hidden">
          <div className="border-b border-slate-100 bg-slate-50/60 px-5 py-3 sm:px-6">
            <h3 className="text-sm font-semibold text-slate-900">Grounded answer</h3>
            <p className="mt-0.5 text-xs text-slate-500">
              Wording reproduced from validated model output; citations below.
            </p>
          </div>
          <div className="max-w-read px-5 py-5 sm:px-6 sm:py-6">
            <AnswerContent answer={response.answer} />
          </div>
          {response.generation_metadata && (
            <dl className="grid gap-3 border-t border-slate-100 bg-slate-50/40 px-5 py-4 text-xs sm:grid-cols-3 sm:px-6 max-w-read">
              <Meta label="Evidence chunks" value={response.generation_metadata.evidence_chunk_count} />
              <Meta
                label="Validated citations"
                value={response.generation_metadata.validated_citation_count}
              />
              <Meta
                label="Latency"
                value={`${Math.round(response.generation_metadata.latency_ms)} ms`}
              />
            </dl>
          )}
        </section>
      )}

      {!success && (
        <section className="panel-elevated p-5 sm:p-6">
          <h3 className="text-sm font-semibold text-slate-900">Response details</h3>
          <div className="mt-3 max-w-read space-y-3 text-sm leading-relaxed text-slate-700">
            {response.answer && (
              <AnswerContent answer={response.answer} />
            )}
            {response.clarification_question && (
              <div className="rounded-lg border border-sky-100 bg-sky-50/50 p-4 text-sky-950">
                <p className="text-label text-sky-800/80">Clarification needed</p>
                <p className="mt-2">{response.clarification_question}</p>
              </div>
            )}
            {response.reason && (
              <p className="text-slate-700">{response.reason}</p>
            )}
            {response.abstention_reason && (
              <div className="rounded-lg border border-amber-100 bg-amber-50/40 p-4">
                <p className="text-label text-amber-900/80">Abstention</p>
                <p className="mt-2 text-amber-950">{response.abstention_reason}</p>
              </div>
            )}
            {!response.answer &&
              !response.reason &&
              !response.abstention_reason &&
              !response.clarification_question && (
                <p className="text-slate-500">
                  No additional message was returned for this status.
                </p>
              )}
          </div>
        </section>
      )}

      {success && response.citations.length > 0 && (
        <section className="max-w-read min-w-0 space-y-3">
          <h3 className="text-label">
            Evidence &amp; citations ({response.citations.length})
          </h3>
          <div className="space-y-3">
            {response.citations.map((citation, index) => (
              <CitationCard key={citation.chunk_id} citation={citation} index={index} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className="mt-0.5 font-medium text-slate-800">{value}</dd>
    </div>
  );
}