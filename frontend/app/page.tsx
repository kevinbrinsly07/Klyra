"use client";

import { useEffect, useState } from "react";

type ChatSource = {
  filename: string;
  document_id?: string | null;
  distance?: number | null;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
};

type ConversationSummary = {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message: string | null;
};

type DocumentItem = {
  document_id: string;
  filename: string;
  total_chunks: number;
};

export default function Home() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [view, setView] = useState<"chat" | "documents">("chat");

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");

  async function handleDocumentUpload(file: File) {
    setUploading(true);
    setUploadError("");
    setUploadSuccess("");

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiUrl}/api/v1/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Document upload failed.");
      }

      setUploadSuccess(
        `${data.filename} uploaded successfully. ${data.total_chunks} chunks indexed.`,
      );

      await refreshDocuments();
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "Document upload failed.",
      );
    } finally {
      setUploading(false);
    }
  }

  async function handleDeleteDocument(documentId: string) {
    const confirmed = window.confirm(
      "Are you sure you want to delete this document? This action cannot be undone.",
    );

    if (!confirmed) return;

    setUploadError("");
    setUploadSuccess("");

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

      const response = await fetch(`${apiUrl}/api/v1/documents/${documentId}`, {
        method: "DELETE",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Document deletion failed.");
      }

      setUploadSuccess("Document deleted successfully.");

      await refreshDocuments();
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "Document deletion failed.",
      );
    }
  }

  useEffect(() => {
    async function loadDocuments() {
      try {
        const apiUrl =
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

        const response = await fetch(
          `${apiUrl}/api/v1/documents/?limit=100&offset=0`,
        );

        if (!response.ok) {
          throw new Error("Failed to load documents");
        }

        const data = await response.json();
        setDocuments(data.documents);
      } catch (err) {
        console.error("Document loading error:", err);
      }
    }

    loadDocuments();
  }, []);

  async function refreshDocuments() {
    setDocumentsLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

      const response = await fetch(
        `${apiUrl}/api/v1/documents/?limit=100&offset=0`,
      );

      if (!response.ok) {
        throw new Error("Failed to load documents");
      }

      const data = await response.json();
      setDocuments(data.documents);
    } catch (err) {
      console.error("Document loading error:", err);
    } finally {
      setDocumentsLoading(false);
    }
  }

  async function refreshConversations() {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

      const response = await fetch(
        `${apiUrl}/api/v1/conversations?limit=20&offset=0`,
      );

      if (!response.ok) {
        throw new Error("Failed to load conversations");
      }

      const data = await response.json();
      setConversations(data.conversations);
    } catch (err) {
      console.error("Conversation loading error:", err);
    }
  }

  useEffect(() => {
    async function loadConversations() {
      try {
        const apiUrl =
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

        const response = await fetch(
          `${apiUrl}/api/v1/conversations?limit=20&offset=0`,
        );

        if (!response.ok) {
          throw new Error("Failed to load conversations");
        }

        const data = await response.json();
        setConversations(data.conversations);
      } catch (err) {
        console.error("Conversation loading error:", err);
      }
    }

    loadConversations();
  }, []);

  async function handleSelectConversation(id: string) {
    if (loading) return;

    setError("");
    setLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

      const response = await fetch(
        `${apiUrl}/api/v1/conversations/${id}/messages?limit=100&offset=0`,
      );

      if (!response.ok) {
        throw new Error("Failed to load conversation history");
      }

      const data = await response.json();

      setConversationId(data.conversation_id);

      setMessages(
        data.messages.map(
          (item: { role: "user" | "assistant"; content: string }) => ({
            role: item.role,
            content: item.content,
          }),
        ),
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load conversation.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleSend() {
    const trimmed = message.trim();

    if (!trimmed || loading) return;

    setMessage("");
    setError("");

    setMessages((current) => [...current, { role: "user", content: trimmed }]);

    setLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

      const response = await fetch(`${apiUrl}/api/v1/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: trimmed,
          top_k: 3,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || `Request failed (${response.status})`,
        );
      }

      const data = await response.json();

      setConversationId(data.conversation_id);
      await refreshConversations();

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.response,
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  function handleNewChat() {
    setMessages([]);
    setMessage("");
    setConversationId(null);
    setError("");
    setView("chat");
  }

  return (
    <main className="flex h-screen overflow-hidden bg-white text-zinc-900">
      <aside className="hidden w-72 shrink-0 flex-col border-r border-zinc-800 bg-[#171717] text-white md:flex">
        <div className="flex items-center gap-3 px-6 py-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white text-lg font-bold text-black">
            K
          </div>
          <span className="text-xl font-semibold tracking-tight">Klyra</span>
        </div>

        <div className="px-4">
          <button
            onClick={handleNewChat}
            className="flex w-full items-center gap-3 rounded-xl border border-zinc-700 px-4 py-3 text-sm font-medium transition hover:bg-zinc-800"
          >
            <span className="text-xl">+</span>
            New chat
          </button>
        </div>

        <div className="mt-8 px-5">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Workspace
          </p>
          <button
            onClick={() => {
              setView("documents");
              void refreshDocuments();
            }}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm ${
              view === "documents"
                ? "bg-zinc-800 text-white"
                : "text-zinc-300 hover:bg-zinc-800"
            }`}
          >
            <span>▤</span>
            Documents
          </button>
          <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm text-zinc-300 hover:bg-zinc-800">
            <span>⌘</span>
            Search
          </button>
        </div>

        <div className="mt-8 flex-1 overflow-y-auto px-5">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Recent chats
          </p>

          {conversations.length > 0 ? (
            <div className="space-y-1">
              {conversations.map((conversation) => (
                <button
                  key={conversation.id}
                  onClick={() => handleSelectConversation(conversation.id)}
                  disabled={loading}
                  className={`w-full truncate rounded-lg px-3 py-2 text-left text-sm transition hover:bg-zinc-800 disabled:cursor-not-allowed ${
                    conversationId === conversation.id
                      ? "bg-zinc-800 text-white"
                      : "text-zinc-300"
                  }`}
                >
                  {conversation.title ||
                    conversation.last_message ||
                    "New conversation"}
                </button>
              ))}
            </div>
          ) : (
            <p className="px-3 py-2 text-sm text-zinc-600">
              Your conversations will appear here.
            </p>
          )}
        </div>

        <div className="border-t border-zinc-800 p-4">
          <div className="flex items-center gap-3 rounded-xl px-2 py-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-zinc-700 text-sm font-semibold">
              K
            </div>
            <div>
              <p className="text-sm font-medium">My workspace</p>
              <p className="text-xs text-zinc-500">Personal account</p>
            </div>
          </div>
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-zinc-100 px-5 md:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-900 font-bold text-white md:hidden">
              K
            </div>
            <span className="text-sm font-semibold">
              {view === "documents"
                ? "Documents"
                : conversationId
                  ? "Conversation"
                  : "New conversation"}
            </span>
          </div>
          <span className="rounded-full border border-zinc-200 px-3 py-1 text-xs text-zinc-500">
            AI Assistant
          </span>
        </header>

        <div className="flex flex-1 flex-col overflow-y-auto">
          {view === "documents" ? (
            <div className="mx-auto w-full max-w-4xl px-6 py-10">
              <div className="mb-8">
                <h1 className="text-2xl font-semibold tracking-tight">
                  Your documents
                </h1>
                <p className="mt-2 text-sm text-zinc-500">
                  Manage the documents in your Klyra knowledge base.
                </p>

                <div className="mt-6 rounded-xl border border-zinc-200 bg-zinc-50 p-5">
                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div>
                      <p className="text-sm font-medium text-zinc-800">
                        Upload a PDF document
                      </p>
                      <p className="mt-1 text-xs text-zinc-500">
                        Maximum file size: 10 MB
                      </p>
                    </div>

                    <label className="cursor-pointer rounded-lg bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-700">
                      {uploading ? "Uploading..." : "Choose PDF"}
                      <input
                        type="file"
                        accept=".pdf,application/pdf"
                        disabled={uploading}
                        className="hidden"
                        onChange={(event) => {
                          const file = event.target.files?.[0];

                          if (file) {
                            void handleDocumentUpload(file);
                          }

                          event.target.value = "";
                        }}
                      />
                    </label>
                  </div>

                  {uploadError && (
                    <p className="mt-3 text-sm text-red-600">{uploadError}</p>
                  )}

                  {uploadSuccess && (
                    <p className="mt-3 text-sm text-green-700">
                      {uploadSuccess}
                    </p>
                  )}
                </div>
              </div>

              <div className="mb-6 flex items-center justify-between">
                <p className="text-sm font-medium text-zinc-700">
                  {documents.length} documents
                </p>

                <button
                  onClick={() => void refreshDocuments()}
                  disabled={documentsLoading}
                  className="rounded-lg border border-zinc-200 px-3 py-2 text-sm hover:bg-zinc-50 disabled:opacity-50"
                >
                  Refresh
                </button>
              </div>

              {documentsLoading ? (
                <p className="py-10 text-center text-sm text-zinc-500">
                  Loading documents...
                </p>
              ) : documents.length === 0 ? (
                <div className="rounded-xl border border-dashed border-zinc-300 px-6 py-16 text-center">
                  <p className="text-sm font-medium text-zinc-700">
                    No documents uploaded yet
                  </p>
                  <p className="mt-2 text-sm text-zinc-500">
                    Upload a PDF to start building your knowledge base.
                  </p>
                </div>
              ) : (
                <div className="divide-y divide-zinc-100 rounded-xl border border-zinc-200">
                  {documents.map((document) => (
                    <div
                      key={document.document_id}
                      className="flex items-center justify-between gap-4 p-4"
                    >
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-zinc-800">
                          {document.filename}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                          {document.total_chunks} indexed chunks
                        </p>
                      </div>

                      <div className="flex shrink-0 items-center gap-3">
                        <span className="rounded-full bg-zinc-100 px-3 py-1 text-xs text-zinc-600">
                          PDF
                        </span>

                        <button
                          onClick={() =>
                            void handleDeleteDocument(document.document_id)
                          }
                          className="rounded-lg px-3 py-1.5 text-xs font-medium text-red-600 transition hover:bg-red-50"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center px-6 pb-16 text-center">
              <div className="mb-7 flex h-16 w-16 items-center justify-center rounded-2xl bg-zinc-900 text-3xl font-bold text-white shadow-lg">
                K
              </div>
              <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
                What can I help you with?
              </h1>
              <p className="mt-3 max-w-md text-sm leading-6 text-zinc-500">
                Ask questions about your documents and get clear, context-aware
                answers powered by your knowledge base.
              </p>

              <div className="mt-10 grid w-full max-w-2xl gap-3 sm:grid-cols-2">
                {[
                  {
                    title: "Explore my documents",
                    description: "What information is available?",
                  },
                  {
                    title: "Summarize a topic",
                    description: "Give me the key points.",
                  },
                  {
                    title: "Find specific information",
                    description: "Search across my knowledge base.",
                  },
                  {
                    title: "Ask a follow-up",
                    description: "Continue where I left off.",
                  },
                ].map((item) => (
                  <button
                    key={item.title}
                    onClick={() => setMessage(item.description)}
                    className="rounded-xl border border-zinc-200 p-4 text-left transition hover:border-zinc-400 hover:bg-zinc-50"
                  >
                    <p className="text-sm font-semibold">{item.title}</p>
                    <p className="mt-1 text-xs text-zinc-500">
                      {item.description}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-7 px-5 py-10">
              {messages.map((item, index) => (
                <div
                  key={index}
                  className={`flex gap-4 ${
                    item.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {item.role === "assistant" && (
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-900 text-sm font-bold text-white">
                      K
                    </div>
                  )}

                  <div className="max-w-[85%]">
                    <div
                      className={`whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-7 ${
                        item.role === "user"
                          ? "bg-zinc-900 text-white"
                          : "bg-zinc-100 text-zinc-800"
                      }`}
                    >
                      {item.content}
                    </div>

                    {item.role === "assistant" &&
                      item.sources &&
                      item.sources.length > 0 && (
                        <div className="mt-3 px-1">
                          <p className="mb-2 text-xs font-semibold text-zinc-500">
                            Sources
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {item.sources.map((source, sourceIndex) => (
                              <span
                                key={`${source.filename}-${sourceIndex}`}
                                className="rounded-lg border border-zinc-200 px-3 py-1.5 text-xs text-zinc-600"
                              >
                                {source.filename}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-900 text-sm font-bold text-white">
                    K
                  </div>
                  <div className="rounded-2xl bg-zinc-100 px-4 py-3 text-sm text-zinc-500">
                    Thinking...
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        {view === "chat" && (
          <div className="shrink-0 px-4 pb-5 pt-3 sm:px-8">
            {error && (
              <p className="mx-auto mb-3 max-w-3xl rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600">
                {error}
              </p>
            )}

            <form
              onSubmit={(event) => {
                event.preventDefault();
                handleSend();
              }}
              className="mx-auto flex max-w-3xl items-end gap-3 rounded-2xl border border-zinc-200 bg-white p-3 shadow-sm transition focus-within:border-zinc-400 focus-within:shadow-md"
            >
              <button
                type="button"
                aria-label="Attach a document"
                title="Document upload coming soon"
                className="mb-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-xl text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700"
              >
                +
              </button>

              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask anything about your documents..."
                rows={1}
                disabled={loading}
                className="max-h-32 min-h-9 flex-1 resize-none bg-transparent py-2 text-sm outline-none placeholder:text-zinc-400"
              />

              <button
                type="submit"
                disabled={!message.trim() || loading}
                aria-label="Send message"
                className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-zinc-900 text-white transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-400"
              >
                ↑
              </button>
            </form>
            <p className="mt-3 text-center text-[11px] text-zinc-400">
              Klyra can make mistakes. Verify important information with your
              source documents.
            </p>
          </div>
        )}
      </section>
    </main>
  );
}
