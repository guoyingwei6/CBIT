import { analyzeGbcLocally } from "./gbcLocalAnalysis.js";

const MAX_FILE_SIZE = 500 * 1024 * 1024;
const COMPUTE_API = "/compute-api";

function apiUrl(path) {
  return `${COMPUTE_API}${path}`;
}

async function responseJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return body;
}

export async function uploadGbcFile(file) {
  if (!(file instanceof File)) {
    throw new Error("Please choose a genotype file");
  }
  if (file.size > MAX_FILE_SIZE) {
    throw new Error("The genotype file exceeds the 500MB limit");
  }

  const target = await responseJson(
    await fetch(apiUrl("/api/uploads/presign"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fileName: file.name,
        size: file.size,
        contentType: file.type || "text/plain",
      }),
    })
  );
  const uploadUrl = /^https?:\/\//.test(target.uploadUrl)
    ? target.uploadUrl
    : apiUrl(target.uploadUrl);
  const uploadResponse = await fetch(uploadUrl, {
    method: "PUT",
    headers: target.headers,
    body: file,
  });
  if (!uploadResponse.ok) {
    throw new Error(`File upload failed (${uploadResponse.status})`);
  }
  return target.objectKey;
}

export async function analyzeGbcObject(objectKey, threshold) {
  return responseJson(
    await fetch(apiUrl("/api/mongo/gbc-estimator/"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ objectKey, threshold }),
    })
  );
}

function shouldUseCloudFallback(error) {
  return error && ["LOCAL_UNAVAILABLE", "LOCAL_FAILURE"].includes(error.code);
}

export async function analyzeGbcFile(file, threshold) {
  if (!(file instanceof File)) {
    throw new Error("Please choose a genotype file");
  }
  if (file.size > MAX_FILE_SIZE) {
    throw new Error("The genotype file exceeds the 500MB limit");
  }
  try {
    return await analyzeGbcLocally(file, threshold);
  } catch (error) {
    if (!shouldUseCloudFallback(error)) {
      throw error;
    }
  }

  const objectKey = await uploadGbcFile(file);
  const result = await analyzeGbcObject(objectKey, threshold);
  return {
    ...result,
    metrics: { ...(result.metrics || {}), execution: "cloud" },
  };
}

export function gbcResultDownloadUrl(csv) {
  return URL.createObjectURL(
    new Blob([csv], { type: "text/csv;charset=utf-8" })
  );
}

export function releaseGbcDownloadUrl(url) {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}
