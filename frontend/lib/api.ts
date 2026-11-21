const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL && process.env.NODE_ENV === "production") {
  throw new Error("NEXT_PUBLIC_API_URL is not configured.");
}

export const API_BASE_URL = (
  API_URL || "http://localhost:8000"
).replace(/\/+$/, "");