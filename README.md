# CBIT Serverless Deployment

This branch is a deployment-only snapshot. It contains a dependency-free static
frontend, browser model assets, the thin GBC fallback service, and deployment files. The
legacy Django application, SMTP configuration, MongoDB, Redis, Celery, uWSGI,
and server scripts are deliberately excluded.

## Architecture

The frontend UI stays on Cloudflare Pages. It uses native HTML, CSS, ES modules,
Canvas, SVG, and Web Workers; no frontend framework or charting runtime is shipped.
Breed identification runs entirely
in the browser with ONNX Runtime Web. GBC also runs in a browser Web Worker by
default, using a small WebAssembly SIMD kernel and a compressed reference asset
served by Pages. The genotype file remains on the user's device on this path.

Cloud Run is a compatibility and resource fallback only. Unsupported browsers,
low-memory/mobile devices, local timeouts, singular matrices, and files above
the local 100-sample limit use the existing presigned R2 upload flow.

```text
Browser -- UI / ONNX / GBC assets --> Cloudflare Pages
Browser -- genotype ----------------> Web Worker + WebAssembly (default)

Fallback only:
Browser -- presigned PUT -----------> Cloudflare R2
Pages Function -- gateway secret ---> Cloud Run GBC service
Cloud Run -- streamed input --------> Cloudflare R2
```

Cloudflare Workers do not run either model. Cloud Run is configured with one
instance at most, so simultaneous fallback requests queue instead of
multiplying memory and CPU usage.

## 0. Runtime Assets

The browser GBC reference and WebAssembly kernel are committed under
`app.bak/public/gbc`. The Cloud Run image reuses that same compressed reference,
verifies both compressed and payload SHA-256 checksums, and memory-maps the
unpacked binary. No duplicate model matrix is stored in this branch.

The deployment branch therefore does not need the legacy pickle, joblib,
Django source tree, or a data-preparation environment. Reference regeneration
belongs on the development branch, where the source matrix is maintained.

## 1. Prepare Cloudflare R2

Create an R2 bucket named `cbit-uploads` and an R2 API token limited to object
read/write access for that bucket. Record the account-specific S3 endpoint,
access key ID, and secret access key in a password manager or secret store.

Apply direct-upload CORS from the repository root:

```bash
npx wrangler r2 bucket cors set cbit-uploads --file deploy/r2-cors.json
npx wrangler r2 bucket cors list cbit-uploads
```

Add a lifecycle safety net for uploads that are not deleted after processing:

```bash
npx wrangler r2 bucket lifecycle add cbit-uploads expire-uploads uploads/ --expire-days 1
```

Add the actual Pages production/preview origins to `r2-cors.json` before
deploying if they differ from `https://cattlebreeds.tech`.

## 2. Build The GBC Container

Enable Cloud Build, Artifact Registry, Cloud Run, and Secret Manager in the
Google Cloud project. Create the Artifact Registry repository once:

```bash
gcloud artifacts repositories create cbit \
  --repository-format=docker \
  --location=asia-east1
```

Build with a revision-specific tag:

```bash
TAG=$(git rev-parse --short HEAD)
gcloud builds submit \
  --config deploy/cloudbuild-gbc.yaml \
  --substitutions "_TAG=${TAG}" \
  .
```

The Docker build unpacks the checked-in compressed reference into compact,
memory-mapped NumPy assets. Only the FastAPI runtime dependencies are installed
in the image.

## 3. Configure Secrets

Create these Secret Manager entries without putting values in Git or shell
history:

- `cbit-r2-access-key-id`
- `cbit-r2-secret-access-key`
- `cbit-gateway-secret`

The gateway secret is one random value shared only by Cloud Run and the Pages
Function. Copy `deploy/cloud-run.env.yaml.example` outside the repository,
replace `YOUR_ACCOUNT_ID`, and keep the resulting file free of secrets.

## 4. Deploy Cloud Run

```bash
PROJECT_ID=$(gcloud config get-value project)
TAG=$(git rev-parse --short HEAD)
IMAGE="asia-east1-docker.pkg.dev/${PROJECT_ID}/cbit/gbc:${TAG}"

gcloud run deploy cbit-gbc \
  --image "${IMAGE}" \
  --region asia-east1 \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 512Mi \
  --concurrency 1 \
  --min 0 \
  --min-instances 0 \
  --max 1 \
  --max-instances 1 \
  --timeout 300 \
  --cpu-throttling \
  --no-cpu-boost \
  --env-vars-file /ABSOLUTE/PATH/cloud-run.env.yaml \
  --set-secrets \
R2_ACCESS_KEY_ID=cbit-r2-access-key-id:latest,R2_SECRET_ACCESS_KEY=cbit-r2-secret-access-key:latest,GATEWAY_SECRET=cbit-gateway-secret:latest
```

Cloud Run must accept unauthenticated network traffic because the Pages
Function does not mint Google IAM identity tokens. All endpoints except
`/healthz` still require the independent `X-CBIT-Gateway` secret.

After deployment, check the reference model loaded correctly:

```bash
curl "$(gcloud run services describe cbit-gbc \
  --region asia-east1 \
  --format 'value(status.url)')/healthz"
```

## 5. Deploy Cloudflare Pages

Use `app.bak` as the Pages project root:

- Build command: `npm ci && npm run build`
- Build output directory: `dist`
- Node.js: 20 or newer

Set these Pages variables for both Production and Preview:

- `GBC_API_URL`: the Cloud Run service URL
- `GBC_GATEWAY_SECRET`: an encrypted secret matching `cbit-gateway-secret`

The frontend calls `/compute-api/...` only after local analysis is unavailable.
The Pages Function adds the gateway secret and forwards only `/api/...`
requests. Genotype files never pass through the Function.

## 6. Cost And Acceptance Checks

Create Google Cloud budget notifications at a low amount such as USD 1 and USD
5. Budget notifications are alerts, not a hard spending cap; the Cloud Run
maximum instance setting is the primary compute bound.

Before switching DNS, verify:

1. Both Breed modes return the same breed labels and probability error stays at
   or below `1e-6` on the regression fixtures.
2. GBC output matches the legacy implementation at four decimal places.
3. A browser `100 x 200,000` benchmark finishes within 30 seconds and WebAssembly
   memory remains below 256 MiB.
4. Files with more than the configured `MAX_SAMPLES` value are rejected before
   the GBC batch matrix is allocated.
5. Chrome, Edge, Firefox, and Safari desktop can compute locally, paginate the
   table, and download the result. Unsupported environments automatically use
   R2 and Cloud Run.
6. Cloud Run shows zero idle instances and never scales above one instance.

## Local Validation

The service defaults to local disk storage. Unpack the checked-in reference and
run it without R2:

```bash
python services/gbc/scripts/prepare_reference.py \
  app.bak/public/gbc \
  /tmp/cbit-gbc-assets

python -m venv /tmp/cbit-gbc-runtime
/tmp/cbit-gbc-runtime/bin/pip install -r services/gbc/requirements.txt
GBC_ASSET_DIR=/tmp/cbit-gbc-assets \
  /tmp/cbit-gbc-runtime/bin/uvicorn services.gbc.app.main:app \
  --host 127.0.0.1 \
  --port 8081
```

Run the numerical, desktop-browser, real UI, and responsive-layout checks:

```bash
cd app.bak
npm run build
npm run test:breed-models
npm run test:gbc-browser
npm run serve:gbc-benchmark -- \
  --genotype /tmp/cbit-gbc-200k-100.txt \
  --fixture /tmp/cbit-gbc-200k-100.json
npm run test:gbc-browsers -- --url http://127.0.0.1:4175/
npm run test:home-ui -- --url http://127.0.0.1:8080/#/home
npm run test:breed-ui -- --url http://127.0.0.1:8080/#/Breed_identification
npm run test:gbc-ui -- --url http://127.0.0.1:8080/#/GBC_estimation
npm run test:responsive-ui -- --url http://127.0.0.1:8080
```

Safari WebDriver validation requires **Allow remote automation** in Safari's
Developer settings. This setting is deliberately not changed by the test
suite.
