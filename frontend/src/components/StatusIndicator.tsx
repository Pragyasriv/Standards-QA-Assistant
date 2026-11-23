import type { ReactNode } from "react";
import { Database, Loader2, Server } from "lucide-react";
import type { ApiStatusState } from "../hooks/useApiStatus";

type Props = {
  status: ApiStatusState;
};

function Dot({ color }: { color: string }) {
  return (
    <span
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${color}`}
      aria-hidden
    />
  );
}

export function StatusIndicator({ status }: Props) {
  const { loading, health, readiness, healthError, readinessError } = status;

  if (loading && !health && !readiness) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-600">
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin text-slate-500" />
          Checking API status…
        </div>
      </div>
    );
  }

  const apiUp = health?.status === "ok" && !healthError;
  const resourcesReady =
    readiness?.status === "ready" &&
    readiness.qdrant_available &&
    readiness.bm25_index_available &&
    !readinessError;

  return (
    <div className="space-y-2 rounded-lg border border-slate-200 bg-white p-3 text-sm">
      <StatusRow
        icon={<Server className="h-4 w-4 text-slate-500" />}
        label="API"
        ok={apiUp}
        detail={
          apiUp
            ? `Running v${health?.version ?? "—"}`
            : (healthError ?? "Unavailable")
        }
      />
      <StatusRow
        icon={<Database className="h-4 w-4 text-slate-500" />}
        label="Readiness"
        ok={resourcesReady}
        detail={
          readiness
            ? readiness.status === "ready"
              ? "Qdrant and BM25 index available"
              : `Qdrant ${readiness.qdrant_available ? "OK" : "missing"}, BM25 ${
                  readiness.bm25_index_available ? "OK" : "missing"
                }`
            : (readinessError ?? "Unknown")
        }
      />
      {!apiUp && (
        <p className="text-xs leading-relaxed text-slate-500">
          Start the backend with{" "}
          <code className="rounded bg-slate-100 px-1">uvicorn app.main:app</code> on
          port 8000.
        </p>
      )}
    </div>
  );
}

function StatusRow({
  icon,
  label,
  ok,
  detail,
}: {
  icon: ReactNode;
  label: string;
  ok: boolean;
  detail: string;
}) {
  return (
    <div className="flex items-start gap-2 text-slate-700">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 font-medium text-slate-800">
          {icon}
          <span>{label}</span>
          <Dot color={ok ? "bg-emerald-500" : "bg-amber-500"} />
        </div>
        <p className="mt-0.5 text-xs leading-relaxed text-slate-500">{detail}</p>
      </div>
    </div>
  );
}