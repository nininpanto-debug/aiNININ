# aiNININ Backend

This is the cloud backend for the aiNININ phone/web app.

## Render

Create a **Web Service** from this repository.

Build Command:
`pip install -r requirements.txt`

Start Command:
`uvicorn aininin_backend:app --host 0.0.0.0 --port $PORT`

Environment Variable:
`FAL_KEY` = your Fal API key

**Never put your FAL_KEY inside GitHub files.**

After deployment, open:
`https://YOUR-SERVICE.onrender.com/`

You should see JSON saying the aiNININ backend is online.

The backend uses Fal's current `fal-ai/ip-adapter-face-id` schema with
`face_image_url` for identity reference. It intentionally has no generic
non-identity fallback.
