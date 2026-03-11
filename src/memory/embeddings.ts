import type { OpenClawConfig } from "../config/config.js";
import type { SecretInput } from "../config/types.secrets.js";
import { formatErrorMessage } from "../infra/errors.js";
import { createGeminiEmbeddingProvider, type GeminiEmbeddingClient } from "./embeddings-gemini.js";
import {
  createMistralEmbeddingProvider,
  type MistralEmbeddingClient,
} from "./embeddings-mistral.js";
import { createOllamaEmbeddingProvider, type OllamaEmbeddingClient } from "./embeddings-ollama.js";
import { createOpenAiEmbeddingProvider, type OpenAiEmbeddingClient } from "./embeddings-openai.js";
import { createVoyageEmbeddingProvider, type VoyageEmbeddingClient } from "./embeddings-voyage.js";

export type { GeminiEmbeddingClient } from "./embeddings-gemini.js";
export type { MistralEmbeddingClient } from "./embeddings-mistral.js";
export type { OpenAiEmbeddingClient } from "./embeddings-openai.js";
export type { VoyageEmbeddingClient } from "./embeddings-voyage.js";
export type { OllamaEmbeddingClient } from "./embeddings-ollama.js";

export type EmbeddingProvider = {
  id: string;
  model: string;
  maxInputTokens?: number;
  embedQuery: (text: string) => Promise<number[]>;
  embedBatch: (texts: string[]) => Promise<number[][]>;
};

export type EmbeddingProviderId = "openai" | "gemini" | "voyage" | "mistral" | "ollama";
export type EmbeddingProviderRequest = EmbeddingProviderId | "auto";
export type EmbeddingProviderFallback = EmbeddingProviderId | "none";

// Remote providers considered for auto-selection when provider === "auto".
// Ollama is intentionally excluded here so that "auto" mode does not
// implicitly assume a local Ollama instance is available.
const REMOTE_EMBEDDING_PROVIDER_IDS = ["openai", "gemini", "voyage", "mistral"] as const;

export type EmbeddingProviderResult = {
  provider: EmbeddingProvider | null;
  requestedProvider: EmbeddingProviderRequest;
  fallbackFrom?: EmbeddingProviderId;
  fallbackReason?: string;
  providerUnavailableReason?: string;
  openAi?: OpenAiEmbeddingClient;
  gemini?: GeminiEmbeddingClient;
  voyage?: VoyageEmbeddingClient;
  mistral?: MistralEmbeddingClient;
  ollama?: OllamaEmbeddingClient;
};

export type EmbeddingProviderOptions = {
  config: OpenClawConfig;
  agentDir?: string;
  provider: EmbeddingProviderRequest;
  remote?: {
    baseUrl?: string;
    apiKey?: SecretInput;
    headers?: Record<string, string>;
  };
  model: string;
  fallback: EmbeddingProviderFallback;
};

function isMissingApiKeyError(err: unknown): boolean {
  const message = formatErrorMessage(err);
  return message.includes("No API key found for provider");
}

export async function createEmbeddingProvider(
  options: EmbeddingProviderOptions,
): Promise<EmbeddingProviderResult> {
  const requestedProvider = options.provider;
  const fallback = options.fallback;

  const createProvider = async (id: EmbeddingProviderId) => {
    if (id === "ollama") {
      const { provider, client } = await createOllamaEmbeddingProvider(options);
      return { provider, ollama: client };
    }
    if (id === "gemini") {
      const { provider, client } = await createGeminiEmbeddingProvider(options);
      return { provider, gemini: client };
    }
    if (id === "voyage") {
      const { provider, client } = await createVoyageEmbeddingProvider(options);
      return { provider, voyage: client };
    }
    if (id === "mistral") {
      const { provider, client } = await createMistralEmbeddingProvider(options);
      return { provider, mistral: client };
    }
    const { provider, client } = await createOpenAiEmbeddingProvider(options);
    return { provider, openAi: client };
  };

  const formatPrimaryError = (err: unknown, _provider: EmbeddingProviderId) =>
    formatErrorMessage(err);

  if (requestedProvider === "auto") {
    const missingKeyErrors: string[] = [];

    for (const provider of REMOTE_EMBEDDING_PROVIDER_IDS) {
      try {
        const result = await createProvider(provider);
        return { ...result, requestedProvider };
      } catch (err) {
        const message = formatPrimaryError(err, provider);
        if (isMissingApiKeyError(err)) {
          missingKeyErrors.push(message);
          continue;
        }
        // Non-auth errors (e.g., network) are still fatal
        const wrapped = new Error(message) as Error & { cause?: unknown };
        wrapped.cause = err;
        throw wrapped;
      }
    }

    // All providers failed due to missing API keys - return null provider for FTS-only mode
    const reason =
      missingKeyErrors.length > 0
        ? missingKeyErrors.join("\n\n")
        : "No embeddings provider available.";
    return {
      provider: null,
      requestedProvider,
      providerUnavailableReason: reason,
    };
  }

  try {
    const primary = await createProvider(requestedProvider);
    return { ...primary, requestedProvider };
  } catch (primaryErr) {
    const reason = formatPrimaryError(primaryErr, requestedProvider);
    if (fallback && fallback !== "none" && fallback !== requestedProvider) {
      try {
        const fallbackResult = await createProvider(fallback);
        return {
          ...fallbackResult,
          requestedProvider,
          fallbackFrom: requestedProvider,
          fallbackReason: reason,
        };
      } catch (fallbackErr) {
        // Both primary and fallback failed - check if it's auth-related
        const fallbackReason = formatErrorMessage(fallbackErr);
        const combinedReason = `${reason}\n\nFallback to ${fallback} failed: ${fallbackReason}`;
        if (isMissingApiKeyError(primaryErr) && isMissingApiKeyError(fallbackErr)) {
          // Both failed due to missing API keys - return null for FTS-only mode
          return {
            provider: null,
            requestedProvider,
            fallbackFrom: requestedProvider,
            fallbackReason: reason,
            providerUnavailableReason: combinedReason,
          };
        }
        // Non-auth errors are still fatal
        const wrapped = new Error(combinedReason) as Error & { cause?: unknown };
        wrapped.cause = fallbackErr;
        throw wrapped;
      }
    }
    // No fallback configured - check if we should degrade to FTS-only
    if (isMissingApiKeyError(primaryErr)) {
      return {
        provider: null,
        requestedProvider,
        providerUnavailableReason: reason,
      };
    }
    const wrapped = new Error(reason) as Error & { cause?: unknown };
    wrapped.cause = primaryErr;
    throw wrapped;
  }
}
