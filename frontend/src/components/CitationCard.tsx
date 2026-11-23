import { Check, ChevronDown, ChevronUp, Copy, MapPin } from "lucide-react";
import { useState } from "react";
import type { ResolvedCitation } from "../types/api";

type Props = {
  citation: ResolvedCitation;
  index: number;
};

function locationHighlights(citation: ResolvedCitation): string[] {
  const parts: string[] = [];
  if (citation.article) parts.push(`Article ${citation.article}`);
  if (citation.paragraph) parts.push(`Paragraph ${citation.paragraph}`);
  if (citation.subparagraph) parts.push(`(${citation.subparagraph})`);
  if (citation.annex) parts.push(`Annex ${citation.annex}`);
  if (citation.page_start != null) {
    const pages =
      citation.page_end != null && citation.page_end !== citation.page_start
        ? `pp. ${citation.page_start}–${citation.page_end}`
        : `p. ${citation.page_start}`;
    parts.push(pages);
  }
  return parts;
}

export function CitationCard({ citation, index }: Props) {
  const [open, setOpen] = useState(index === 0);
  const [copied, setCopied] = useState(false);
  const locationParts = locationHighlights(citation);

  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(citation.chunk_id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  const title = citation.regulation_number
    ? `${citation.regulation} · ${citation.regulation_number}`
    : citation.regulation;

  return (
    <article className="panel-elevated overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-start justify-between gap-3 px-4 py-4 text-left sm:px-5"
        aria-expanded={open}
      >
        <div className="min-w-0 flex-1">
          <p className="text-base font-semibold tracking-tight text-slate-900">{title}</p>
          {locationParts.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {locationParts.map((part) => (
                <span
                  key={part}
                  className="inline-flex items-center gap-1 rounded-md bg-accent/5 px-2 py-0.5 text-xs font-medium text-accent"
                >
                  <MapPin className="h-3 w-3 opacity-70" aria-hidden />
                  {part}
                </span>
              ))}
            </div>
          )}
        </div>
        {open ? (
          <ChevronUp className="mt-1 h-4 w-4 shrink-0 text-slate-400" />
        ) : (
          <ChevronDown className="mt-1 h-4 w-4 shrink-0 text-slate-400" />
        )}
      </button>
      {open && (
        <div className="border-t border-slate-100 bg-slate-50/30 px-4 py-4 sm:px-5">
          <p className="text-label mb-3">Document metadata</p>
          <dl className="grid gap-3 text-sm sm:grid-cols-2">
            <Field label="Document ID" value={citation.document_id} />
            {citation.source_filename && (
              <Field label="Source file" value={citation.source_filename} wide />
            )}
            {citation.article && <Field label="Article" value={citation.article} />}
            {citation.paragraph && <Field label="Paragraph" value={citation.paragraph} />}
            {citation.subparagraph && (
              <Field label="Subparagraph" value={citation.subparagraph} />
            )}
            {citation.annex && <Field label="Annex" value={citation.annex} />}
            {citation.page_start != null && (
              <Field
                label="Pages"
                value={
                  citation.page_end != null && citation.page_end !== citation.page_start
                    ? `${citation.page_start}–${citation.page_end}`
                    : String(citation.page_start)
                }
              />
            )}
          </dl>
          <div className="mt-4 rounded-lg border border-dashed border-slate-200 bg-white p-3">
            <p className="text-label mb-2">Chunk ID</p>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <code className="min-w-0 flex-1 break-all rounded-md bg-slate-50 px-2.5 py-2 font-mono text-xs text-slate-800">
                {citation.chunk_id}
              </code>
              <button
                type="button"
                onClick={() => void copyId()}
                className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50"
                title="Copy chunk ID"
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-emerald-600" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" />
                    Copy ID
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </article>
  );
}

function Field({
  label,
  value,
  wide,
}: {
  label: string;
  value: string;
  wide?: boolean;
}) {
  return (
    <div className={wide ? "sm:col-span-2" : undefined}>
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="mt-0.5 font-medium text-slate-800">{value}</dd>
    </div>
  );
}