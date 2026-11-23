import { Loader2, Send } from "lucide-react";
import { useEffect, useId } from "react";

type Props = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
  prominent?: boolean;
};

export function QuestionInput({
  value,
  onChange,
  onSubmit,
  loading,
  prominent,
}: Props) {
  const labelId = useId();
  const disabled = loading || !value.trim();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        if (!loading && value.trim()) {
          onSubmit();
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [loading, onSubmit, value]);

  return (
    <section
      className={
        prominent
          ? "panel-elevated border-accent/15 p-5 shadow-md ring-1 ring-accent/10 sm:p-6"
          : "panel-elevated p-4 sm:p-5"
      }
      aria-labelledby={labelId}
    >
      <label
        id={labelId}
        htmlFor="question-field"
        className="mb-2 block text-sm font-semibold text-slate-900"
      >
        Your question
      </label>
      <textarea
        id="question-field"
        rows={prominent ? 5 : 4}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={loading}
        placeholder="e.g. What does RoHS Article 4(1) require?"
        className={`w-full resize-y rounded-lg border bg-white px-3.5 py-3 text-[0.9375rem] leading-relaxed text-slate-900 outline-none transition placeholder:text-slate-400 disabled:opacity-60 ${
          prominent
            ? "border-accent/25 shadow-inner ring-2 ring-accent/5 focus:border-accent focus:ring-accent/20"
            : "border-slate-200 focus:border-accent focus:ring-2 focus:ring-accent/15"
        }`}
      />
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-slate-500">
          <kbd className="rounded border border-slate-200 bg-slate-100 px-1.5 py-0.5 font-mono text-[11px]">
            Ctrl
          </kbd>{" "}
          +{" "}
          <kbd className="rounded border border-slate-200 bg-slate-100 px-1.5 py-0.5 font-mono text-[11px]">
            Enter
          </kbd>{" "}
          to submit
        </p>
        <button
          type="button"
          onClick={onSubmit}
          disabled={disabled}
          className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-accent-dark disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Processing…
            </>
          ) : (
            <>
              <Send className="h-4 w-4" />
              Ask
            </>
          )}
        </button>
      </div>
    </section>
  );
}