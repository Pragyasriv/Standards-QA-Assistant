import { useCallback, useState } from "react";
import { ApiError, postAsk } from "../services/api";
import type { AskResponse } from "../types/api";

export type AskState = {
  loading: boolean;
  response: AskResponse | null;
  error: string | null;
  submit: (question: string) => Promise<void>;
  reset: () => void;
};

export function useAsk(): AskState {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setResponse(null);
    setError(null);
    setLoading(false);
  }, []);

  const submit = useCallback(
    async (question: string) => {
      if (loading) return;
      const trimmed = question.trim();
      if (!trimmed) return;
      setLoading(true);
      setError(null);
      setResponse(null);
      try {
        setResponse(await postAsk({ question: trimmed }));
      } catch (err) {
        setResponse(null);
        if (err instanceof ApiError) {
          setError(err.message);
        } else {
          setError("An unexpected error occurred.");
        }
      } finally {
        setLoading(false);
      }
    },
    [loading],
  );

  return { loading, response, error, submit, reset };
}