export async function onRequest({ request, env, params }) {
  if (!env.GBC_API_URL || !env.GBC_GATEWAY_SECRET) {
    return Response.json(
      { detail: "GBC compute gateway is not configured" },
      { status: 503 }
    );
  }

  const path = Array.isArray(params.path)
    ? params.path.join("/")
    : params.path || "";
  if (!path.startsWith("api/")) {
    return Response.json({ detail: "Not found" }, { status: 404 });
  }

  const incomingUrl = new URL(request.url);
  const upstreamUrl = new URL(env.GBC_API_URL);
  upstreamUrl.pathname = `${upstreamUrl.pathname.replace(/\/$/, "")}/${path}`;
  upstreamUrl.search = incomingUrl.search;

  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.set("X-CBIT-Gateway", env.GBC_GATEWAY_SECRET);
  const upstreamRequest = new Request(upstreamUrl, {
    method: request.method,
    headers,
    body: ["GET", "HEAD"].includes(request.method) ? null : request.body,
    redirect: "manual",
  });
  return fetch(upstreamRequest);
}
