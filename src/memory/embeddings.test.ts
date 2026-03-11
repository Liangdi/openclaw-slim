import { afterEach, describe, expect, it, vi } from "vitest";
import * as authModule from "../agents/model-auth.js";
import { DEFAULT_GEMINI_EMBEDDING_MODEL } from "./embeddings-gemini.js";
import { createEmbeddingProvider } from "./embeddings.js";

vi.mock("../agents/model-auth.js", async () => {
  const { createModelAuthMockModule } = await import("../test-utils/model-auth-mock.js");
  return createModelAuthMockModule();
});

const createFetchMock = () =>
  vi.fn(async (_input?: unknown, _init?: unknown) => ({
    ok: true,
    status: 200,
    json: async () => ({ data: [{ embedding: [1, 2, 3] }] }),
  }));

const createGeminiFetchMock = () =>
  vi.fn(async (_input?: unknown, _init?: unknown) => ({
    ok: true,
    status: 200,
    json: async () => ({ embedding: { values: [1, 2, 3] } }),
  }));

function readFirstFetchRequest(fetchMock: { mock: { calls: unknown[][] } }) {
  const [url, init] = fetchMock.mock.calls[0] ?? [];
  return { url, init: init as RequestInit | undefined };
}

afterEach(() => {
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

function requireProvider(result: Awaited<ReturnType<typeof createEmbeddingProvider>>) {
  if (!result.provider) {
    throw new Error("Expected embedding provider");
  }
  return result.provider;
}

function mockResolvedProviderKey(apiKey = "provider-key") {
  vi.mocked(authModule.resolveApiKeyForProvider).mockResolvedValue({
    apiKey,
    mode: "api-key",
    source: "test",
  });
}

function expectAutoSelectedProvider(
  result: Awaited<ReturnType<typeof createEmbeddingProvider>>,
  expectedId: "openai" | "gemini" | "mistral",
) {
  expect(result.requestedProvider).toBe("auto");
  const provider = requireProvider(result);
  expect(provider.id).toBe(expectedId);
  return provider;
}

function createAutoProvider(model = "") {
  return createEmbeddingProvider({
    config: {} as never,
    provider: "auto",
    model,
    fallback: "none",
  });
}

describe("embedding provider remote overrides", () => {
  it("uses remote baseUrl/apiKey and merges headers", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    mockResolvedProviderKey("provider-key");

    const cfg = {
      models: {
        providers: {
          openai: {
            baseUrl: "https://api.openai.com/v1",
            headers: {
              "X-Provider": "p",
              "X-Shared": "provider",
            },
          },
        },
      },
    };

    const result = await createEmbeddingProvider({
      config: cfg as never,
      provider: "openai",
      remote: {
        baseUrl: "https://example.com/v1",
        apiKey: "  remote-key  ",
        headers: {
          "X-Shared": "remote",
          "X-Remote": "r",
        },
      },
      model: "text-embedding-3-small",
      fallback: "openai",
    });

    const provider = requireProvider(result);
    await provider.embedQuery("hello");

    expect(authModule.resolveApiKeyForProvider).not.toHaveBeenCalled();
    const url = fetchMock.mock.calls[0]?.[0];
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(url).toBe("https://example.com/v1/embeddings");
    const headers = (init?.headers ?? {}) as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer remote-key");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers["X-Provider"]).toBe("p");
    expect(headers["X-Shared"]).toBe("remote");
    expect(headers["X-Remote"]).toBe("r");
  });

  it("falls back to resolved api key when remote apiKey is blank", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    mockResolvedProviderKey("provider-key");

    const cfg = {
      models: {
        providers: {
          openai: {
            baseUrl: "https://api.openai.com/v1",
          },
        },
      },
    };

    const result = await createEmbeddingProvider({
      config: cfg as never,
      provider: "openai",
      remote: {
        baseUrl: "https://example.com/v1",
        apiKey: "   ",
      },
      model: "text-embedding-3-small",
      fallback: "openai",
    });

    const provider = requireProvider(result);
    await provider.embedQuery("hello");

    expect(authModule.resolveApiKeyForProvider).toHaveBeenCalledTimes(1);
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    const headers = (init?.headers as Record<string, string>) ?? {};
    expect(headers.Authorization).toBe("Bearer provider-key");
  });

  it("builds Gemini embeddings requests with api key header", async () => {
    const fetchMock = createGeminiFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    mockResolvedProviderKey("provider-key");

    const cfg = {
      models: {
        providers: {
          google: {
            baseUrl: "https://generativelanguage.googleapis.com/v1beta",
          },
        },
      },
    };

    const result = await createEmbeddingProvider({
      config: cfg as never,
      provider: "gemini",
      remote: {
        apiKey: "gemini-key",
      },
      model: "text-embedding-004",
      fallback: "openai",
    });

    const provider = requireProvider(result);
    await provider.embedQuery("hello");

    const { url, init } = readFirstFetchRequest(fetchMock);
    expect(url).toBe(
      "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent",
    );
    const headers = (init?.headers ?? {}) as Record<string, string>;
    expect(headers["x-goog-api-key"]).toBe("gemini-key");
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("fails fast when Gemini remote apiKey is an unresolved SecretRef", async () => {
    await expect(
      createEmbeddingProvider({
        config: {} as never,
        provider: "gemini",
        remote: {
          apiKey: { source: "env", provider: "default", id: "GEMINI_API_KEY" },
        },
        model: "text-embedding-004",
        fallback: "openai",
      }),
    ).rejects.toThrow(/agents\.\*\.memorySearch\.remote\.apiKey:/i);
  });

  it("uses GEMINI_API_KEY env indirection for Gemini remote apiKey", async () => {
    const fetchMock = createGeminiFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubEnv("GEMINI_API_KEY", "env-gemini-key");

    const result = await createEmbeddingProvider({
      config: {} as never,
      provider: "gemini",
      remote: {
        apiKey: "GEMINI_API_KEY", // pragma: allowlist secret
      },
      model: "text-embedding-004",
      fallback: "openai",
    });

    const provider = requireProvider(result);
    await provider.embedQuery("hello");

    const { init } = readFirstFetchRequest(fetchMock);
    const headers = (init?.headers ?? {}) as Record<string, string>;
    expect(headers["x-goog-api-key"]).toBe("env-gemini-key");
  });

  it("builds Mistral embeddings requests with bearer auth", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    mockResolvedProviderKey("provider-key");

    const cfg = {
      models: {
        providers: {
          mistral: {
            baseUrl: "https://api.mistral.ai/v1",
          },
        },
      },
    };

    const result = await createEmbeddingProvider({
      config: cfg as never,
      provider: "mistral",
      remote: {
        apiKey: "mistral-key", // pragma: allowlist secret
      },
      model: "mistral/mistral-embed",
      fallback: "none",
    });

    const provider = requireProvider(result);
    await provider.embedQuery("hello");

    const { url, init } = readFirstFetchRequest(fetchMock);
    expect(url).toBe("https://api.mistral.ai/v1/embeddings");
    const headers = (init?.headers ?? {}) as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer mistral-key");
    const payload = JSON.parse((init?.body as string | undefined) ?? "{}") as { model?: string };
    expect(payload.model).toBe("mistral-embed");
  });
});

describe("embedding provider auto selection", () => {
  it("prefers openai when a key resolves", async () => {
    vi.mocked(authModule.resolveApiKeyForProvider).mockImplementation(async ({ provider }) => {
      if (provider === "openai") {
        return { apiKey: "openai-key", source: "env: OPENAI_API_KEY", mode: "api-key" };
      }
      throw new Error(`No API key found for provider "${provider}".`);
    });

    const result = await createAutoProvider();
    expectAutoSelectedProvider(result, "openai");
  });

  it("uses gemini when openai is missing", async () => {
    const fetchMock = createGeminiFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    vi.mocked(authModule.resolveApiKeyForProvider).mockImplementation(async ({ provider }) => {
      if (provider === "openai") {
        throw new Error('No API key found for provider "openai".');
      }
      if (provider === "google") {
        return { apiKey: "gemini-key", source: "env: GEMINI_API_KEY", mode: "api-key" };
      }
      throw new Error(`Unexpected provider ${provider}`);
    });

    const result = await createAutoProvider();
    const provider = expectAutoSelectedProvider(result, "gemini");
    await provider.embedQuery("hello");
    const [url] = fetchMock.mock.calls[0] ?? [];
    expect(url).toBe(
      `https://generativelanguage.googleapis.com/v1beta/models/${DEFAULT_GEMINI_EMBEDDING_MODEL}:embedContent`,
    );
  });

  it("keeps explicit model when openai is selected", async () => {
    const fetchMock = vi.fn(async (_input?: unknown, _init?: unknown) => ({
      ok: true,
      status: 200,
      json: async () => ({ data: [{ embedding: [1, 2, 3] }] }),
    }));
    vi.stubGlobal("fetch", fetchMock);
    vi.mocked(authModule.resolveApiKeyForProvider).mockImplementation(async ({ provider }) => {
      if (provider === "openai") {
        return { apiKey: "openai-key", source: "env: OPENAI_API_KEY", mode: "api-key" };
      }
      throw new Error(`Unexpected provider ${provider}`);
    });

    const result = await createEmbeddingProvider({
      config: {} as never,
      provider: "auto",
      model: "text-embedding-3-small",
      fallback: "none",
    });

    expect(result.requestedProvider).toBe("auto");
    const provider = requireProvider(result);
    expect(provider.id).toBe("openai");
    await provider.embedQuery("hello");
    const url = fetchMock.mock.calls[0]?.[0];
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(url).toBe("https://api.openai.com/v1/embeddings");
    const payload = JSON.parse(init?.body as string) as { model?: string };
    expect(payload.model).toBe("text-embedding-3-small");
  });

  it("uses mistral when openai/gemini/voyage are missing", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    vi.mocked(authModule.resolveApiKeyForProvider).mockImplementation(async ({ provider }) => {
      if (provider === "mistral") {
        return { apiKey: "mistral-key", source: "env: MISTRAL_API_KEY", mode: "api-key" }; // pragma: allowlist secret
      }
      throw new Error(`No API key found for provider "${provider}".`);
    });

    const result = await createAutoProvider();
    const provider = expectAutoSelectedProvider(result, "mistral");
    await provider.embedQuery("hello");
    const [url] = fetchMock.mock.calls[0] ?? [];
    expect(url).toBe("https://api.mistral.ai/v1/embeddings");
  });
});

describe("FTS-only fallback when no provider available", () => {
  it("returns null provider with reason when auto mode finds no providers", async () => {
    vi.mocked(authModule.resolveApiKeyForProvider).mockRejectedValue(
      new Error('No API key found for provider "openai"'),
    );

    const result = await createEmbeddingProvider({
      config: {} as never,
      provider: "auto",
      model: "",
      fallback: "none",
    });

    expect(result.provider).toBeNull();
    expect(result.requestedProvider).toBe("auto");
    expect(result.providerUnavailableReason).toBeDefined();
    expect(result.providerUnavailableReason).toContain("No API key");
  });

  it("returns null provider when explicit provider fails with missing API key", async () => {
    vi.mocked(authModule.resolveApiKeyForProvider).mockRejectedValue(
      new Error('No API key found for provider "openai"'),
    );

    const result = await createEmbeddingProvider({
      config: {} as never,
      provider: "openai",
      model: "text-embedding-3-small",
      fallback: "none",
    });

    expect(result.provider).toBeNull();
    expect(result.requestedProvider).toBe("openai");
    expect(result.providerUnavailableReason).toBeDefined();
  });

  it("returns null provider when both primary and fallback fail with missing API keys", async () => {
    vi.mocked(authModule.resolveApiKeyForProvider).mockRejectedValue(
      new Error("No API key found for provider"),
    );

    const result = await createEmbeddingProvider({
      config: {} as never,
      provider: "openai",
      model: "text-embedding-3-small",
      fallback: "gemini",
    });

    expect(result.provider).toBeNull();
    expect(result.requestedProvider).toBe("openai");
    expect(result.fallbackFrom).toBe("openai");
    expect(result.providerUnavailableReason).toContain("Fallback to gemini failed");
  });
});
