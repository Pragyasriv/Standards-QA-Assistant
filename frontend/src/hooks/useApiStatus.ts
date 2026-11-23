import { useCallback, useEffect, useState } from "react";
import { ApiError, fetchHealth, fetchReady } from "../services/api";
import type { HealthResponse, ReadinessResponse } from "../types/api";

export type ApiStatusState = {
  loading: boolean;
  health: HealthResponse | null;
  readiness: ReadinessResponse | null;
  healthError: string | null;
  readinessError: string | null;
  refresh: () => Promise<void>;
};

export function useApiStatus(pollMs = 30_000): ApiStatusState {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [readinessError, setReadinessError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    let nextHealthError: string | null = null;
    let nextReadinessError: string | null = null;

    try {
      setHealth(await fetchHealth());
    } catch (err) {
      setHealth(null);
      nextHealthError =
        err instanceof ApiError ? err.message : "Health check failed.";
    }

    try {
      setReadiness(await fetchReady());
    } catch (err) {
      setReadiness(null);
      nextReadinessError =
        err instanceof ApiError ? err.message : "Readiness check failed.";
    }

    setHealthError(nextHealthError);
    setReadinessError(nextReadinessError);
    setLoading(false);
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => {
      void refresh();
    }, pollMs);
    return () => window.clearInterval(id);
  }, [refresh, pollMs]);

  return {
    loading,
    health,
    readiness,
    healthError,
    readinessError,
    refresh,
  };
}