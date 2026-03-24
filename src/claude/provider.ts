import {
  query,
  type SDKMessage,
  type SDKAssistantMessage,
  type SDKResultMessage,
  type SDKUserMessage,
  type SDKSystemMessage,
  type Options,
  type CanUseTool,
  type PermissionResult,
  type Query,
} from "@anthropic-ai/claude-agent-sdk";
import { logger } from "../logger.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface McpServerInfo {
  name: string;
  status: string;
}

export interface QueryOptions {
  prompt: string;
  cwd: string;
  resume?: string;
  /** Continue the most recent conversation instead of starting a new one */
  continueRecent?: boolean;
  model?: string;
  permissionMode?: "default" | "acceptEdits" | "plan";
  images?: Array<{
    type: "image";
    source: { type: "base64"; media_type: string; data: string };
  }>;
  onPermissionRequest?: (toolName: string, toolInput: string) => Promise<boolean>;
}

export interface QueryResult {
  text: string;
  sessionId: string;
  error?: string;
  mcpServers?: McpServerInfo[];
}

// ---------------------------------------------------------------------------
// MCP Status function
// ---------------------------------------------------------------------------

/**
 * Get MCP server status using the SDK's mcpServerStatus() method.
 * This creates a minimal query, calls mcpServerStatus(), then interrupts.
 */
export async function getMcpServerStatus(cwd: string): Promise<McpServerInfo[]> {
  const sdkOptions: Options = {
    cwd,
    permissionMode: "plan", // Use plan mode to prevent any tool execution
    settingSources: ["user", "project"],
  };

  try {
    const q: Query = query({ prompt: " ", options: sdkOptions });

    // Wait for init message then get MCP status
    let mcpStatus: McpServerInfo[] = [];

    // Start consuming messages to trigger init
    const messagePromise = (async () => {
      for await (const message of q) {
        if (message.type === "system" && "subtype" in message && message.subtype === "init") {
          // Now we can call mcpServerStatus
          try {
            const status = await q.mcpServerStatus();
            mcpStatus = status.map(s => ({
              name: s.name,
              status: s.status,
            }));
            logger.debug("Got MCP server status", { count: mcpStatus.length });
          } catch (err) {
            logger.error("Failed to get MCP status", { err });
          }
          // Interrupt the query
          await q.interrupt();
          break;
        }
      }
    })();

    await messagePromise;
    return mcpStatus;
  } catch (err) {
    logger.error("getMcpServerStatus threw", { error: err instanceof Error ? err.message : String(err) });
    return [];
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Extract accumulated text from an SDK assistant message's content blocks.
 */
function extractText(msg: SDKAssistantMessage): string {
  const content = msg.message?.content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((block: any) => block.type === "text")
    .map((block: any) => (block.text as string) ?? "")
    .join("");
}

/**
 * Extract session_id from any SDKMessage that carries one.
 */
function getSessionId(msg: SDKMessage): string | undefined {
  if ("session_id" in msg) {
    return (msg as { session_id: string }).session_id;
  }
  return undefined;
}

/**
 * Build an async iterable yielding a single SDKUserMessage with optional
 * image content blocks.  The session_id is set to "" — the SDK assigns the
 * real session id once the process starts.
 */
async function* singleUserMessage(
  text: string,
  images?: QueryOptions["images"],
): AsyncGenerator<SDKUserMessage, void, unknown> {
  const contentBlocks: Array<{
    type: string;
    text?: string;
    source?: { type: "base64"; media_type: string; data: string };
  }> = [{ type: "text", text }];

  if (images?.length) {
    for (const img of images) {
      contentBlocks.push({ type: "image", source: img.source });
    }
  }

  const msg: SDKUserMessage = {
    type: "user",
    session_id: "",
    parent_tool_use_id: null,
    message: {
      role: "user",
      content: contentBlocks,
    },
  };

  yield msg;
}

// ---------------------------------------------------------------------------
// Core function
// ---------------------------------------------------------------------------

export async function claudeQuery(options: QueryOptions): Promise<QueryResult> {
  const {
    prompt,
    cwd,
    resume,
    continueRecent,
    model,
    permissionMode,
    images,
    onPermissionRequest,
  } = options;

  logger.info("Starting Claude query", {
    cwd,
    model,
    permissionMode,
    resume: !!resume,
    continueRecent: !!continueRecent,
    hasImages: !!images?.length,
  });

  // When images are present we use the multi-content AsyncIterable path;
  // otherwise a plain string is simpler and sufficient.
  const hasImages = images && images.length > 0;
  const promptParam: string | AsyncIterable<SDKUserMessage> = hasImages
    ? singleUserMessage(prompt, images)
    : prompt;

  // --- Build SDK options ---
  const sdkOptions: Options = {
    cwd,
    permissionMode,
    settingSources: ["user", "project"],
  };

  if (model) sdkOptions.model = model;
  if (resume) sdkOptions.resume = resume;
  if (continueRecent) sdkOptions.continue = true;

  // Permission callback — bridges the SDK's CanUseTool to our simpler handler.
  if (onPermissionRequest) {
    const canUseTool: CanUseTool = async (
      toolName: string,
      input: Record<string, unknown>,
    ): Promise<PermissionResult> => {
      const inputStr = JSON.stringify(input);
      logger.info("Permission request from SDK", { toolName });
      try {
        const allowed = await onPermissionRequest(toolName, inputStr);
        if (allowed) {
          return { behavior: "allow", updatedInput: input };
        }
        return {
          behavior: "deny",
          message: "Permission denied by user.",
          interrupt: true,
        };
      } catch (err) {
        logger.error("Permission handler error", { toolName, err });
        return {
          behavior: "deny",
          message: "Permission check failed.",
          interrupt: true,
        };
      }
    };
    sdkOptions.canUseTool = canUseTool;
  }

  // --- Execute query & accumulate output ---
  let sessionId = "";
  const textParts: string[] = [];
  let errorMessage: string | undefined;
  let mcpServers: McpServerInfo[] | undefined;

  try {
    const result = query({ prompt: promptParam, options: sdkOptions });

    for await (const message of result) {
      const sid = getSessionId(message);
      if (sid) sessionId = sid;

      switch (message.type) {
        case "assistant": {
          const text = extractText(message as SDKAssistantMessage);
          if (text) textParts.push(text);
          break;
        }
        case "result": {
          const rm = message as SDKResultMessage;
          if (rm.subtype === "success" && "result" in rm) {
            // The SDK result message carries the final result string.
            // Append only when it adds content not yet seen.
            if (rm.result) {
              const combined = textParts.join("");
              if (!combined.includes(rm.result)) {
                textParts.push(rm.result);
              }
            }
          } else if ("errors" in rm && rm.errors.length > 0) {
            errorMessage = rm.errors.join("; ");
            logger.error("SDK returned error result", { errors: rm.errors });
          }
          break;
        }
        case "system": {
          const sysMsg = message as SDKSystemMessage;
          if (sysMsg.subtype === "init" && sysMsg.mcp_servers) {
            mcpServers = sysMsg.mcp_servers.map(s => ({
              name: s.name,
              status: s.status,
            }));
            logger.debug("Captured MCP servers from init message", {
              count: mcpServers.length,
            });
          }
          break;
        }
        default:
          // tool_progress, auth_status, stream_event, etc. — ignore
          break;
      }
    }
  } catch (err: unknown) {
    errorMessage = err instanceof Error ? err.message : String(err);
    logger.error("Claude query threw", { error: errorMessage });
  }

  const fullText = textParts.join("\n").trim();

  if (!fullText && !errorMessage) {
    errorMessage = "Claude returned an empty response.";
  }

  logger.info("Claude query completed", {
    sessionId,
    textLength: fullText.length,
    hasError: !!errorMessage,
  });

  return {
    text: fullText,
    sessionId,
    error: errorMessage,
    mcpServers,
  };
}
