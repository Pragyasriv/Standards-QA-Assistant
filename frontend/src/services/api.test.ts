import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, postAsk } from "./api";

describe("postAsk", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("maps structured API errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        statusText: "Service Unavailable",
        json: async () => ({
          error: "not_ready",
          message: "The question service is not ready.",
        }),
      }),
    );

    await expect(postAsk({ question: "test" })).rejects.toMatchObject({
      name: "ApiError",
      status: 503,
      message: "The question service is not ready.",
    } satisfies Partial<ApiError>);
  });
});