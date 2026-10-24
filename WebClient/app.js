import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.mjs";

const MODEL_ENDPOINT =
  "https://despro-bidisgn-model-1031455690132.asia-southeast2.run.app/predict";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MAX_HANDS = 2;
const LANDMARKS_PER_HAND = 21;
const FEATURES_PER_LANDMARK = 3;
const FEATURES_PER_FRAME = MAX_HANDS * LANDMARKS_PER_HAND * FEATURES_PER_LANDMARK;
const SEQUENCE_LENGTH = 90;

const cameraBaseUrl = document.querySelector("#cameraBaseUrl");
const canvas = document.querySelector("#landmarkCanvas");
const context = canvas.getContext("2d");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const statusElement = document.querySelector("#status");
const predictionElement = document.querySelector("#prediction");
const progressElement = document.querySelector("#progress");

let landmarker;
let sequence = [];
let lastTimestamp = -1;
let running = false;
let predictionInFlight = false;
let lastProcessedTime = -1;
let captureTimer;

function setStatus(message) {
  statusElement.textContent = message;
}

function normalizeCameraUrl(value) {
  const valueWithoutTrailingSlash = value.trim().replace(/\/+$/, "");
  if (!valueWithoutTrailingSlash) {
    throw new Error("Alamat ESP32-CAM wajib diisi.");
  }

  function normalizeCaptureUrl(value) {
    const valueWithoutTrailingSlash = value.trim().replace(/\/+$/, "");
    let url;
    try {
      url = new URL(valueWithoutTrailingSlash);
    } catch {
      throw new Error("Alamat kamera harus berupa URL, misalnya http://192.168.1.50");
    }
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      throw new Error("Alamat kamera harus menggunakan http atau https.");
    }
    if (url.pathname === "/capture") {
      return url.toString();
    }
    url.pathname = "/capture";
    url.search = "";
    return url.toString();
  }
  let url;
  try {
    url = new URL(valueWithoutTrailingSlash);
  } catch {
    throw new Error("Alamat kamera harus berupa URL, misalnya http://192.168.1.50");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Alamat kamera harus menggunakan http atau https.");
  }
  if (url.pathname !== "/" && url.pathname !== "/stream") {
    throw new Error("Gunakan alamat dasar kamera atau alamat /stream.");
  }
  if (url.pathname === "/stream") {
    return url.toString();
  }
  if (!url.port) {
    url.port = "81";
  }
  url.pathname = "/stream";
  return url.toString();
}

async function createLandmarker() {
  if (landmarker) {
    return landmarker;
  }

  setStatus("Memuat model MediaPipe...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: MAX_HANDS,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  return landmarker;
}

function handSlot(results, index) {
  const category = results.handednesses?.[index]?.[0]?.categoryName?.toLowerCase();
  return category === "left" ? 0 : category === "right" ? 1 : null;
}

function flattenLandmarks(results) {
  const hands = Array.from({ length: MAX_HANDS }, () =>
    new Array(LANDMARKS_PER_HAND * FEATURES_PER_LANDMARK).fill(0)
  );
  const usedSlots = new Set();

  for (let index = 0; index < (results.landmarks?.length ?? 0); index += 1) {
    let slot = handSlot(results, index);
    if (slot === null || usedSlots.has(slot)) {
      slot = usedSlots.has(0) ? 1 : 0;
    }
    if (usedSlots.has(slot)) {
      continue;
    }
    usedSlots.add(slot);

    const landmarks = results.landmarks[index];
    for (let point = 0; point < LANDMARKS_PER_HAND; point += 1) {
      const landmark = landmarks[point];
      hands[slot][point * FEATURES_PER_LANDMARK] = landmark.x;
      hands[slot][point * FEATURES_PER_LANDMARK + 1] = landmark.y;
      hands[slot][point * FEATURES_PER_LANDMARK + 2] = landmark.z;
    }
  }

  return hands.flat();
}

async function sendPrediction(frames) {
  predictionInFlight = true;
  try {
    const response = await fetch(MODEL_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks: frames })
    });
    if (!response.ok) {
      throw new Error(`Cloud Run mengembalikan HTTP ${response.status}.`);
    }
    const result = await response.json();
    predictionElement.textContent =
      `${result.label ?? "Tidak diketahui"} (${result.confidence ?? 0}%)`;
  } finally {
    predictionInFlight = false;
  }
}

async function processFrame(timestamp) {
  if (!running) {
    return;
  }
  const captureUrl = normalizeCaptureUrl(cameraBaseUrl.value);
  try {
    const response = await fetch(`${captureUrl}?t=${Date.now()}`, {
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error(`ESP32-CAM mengembalikan HTTP ${response.status}.`);
    }
    const frame = await createImageBitmap(await response.blob());
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(frame, 0, 0, canvas.width, canvas.height);
    frame.close();

    const currentTimestamp = performance.now();
    if (currentTimestamp > lastTimestamp) {
      lastTimestamp = currentTimestamp;
      const results = landmarker.detectForVideo(canvas, currentTimestamp);
      context.fillStyle = "#00ff66";
      for (const hand of results.landmarks ?? []) {
        for (const landmark of hand) {
          context.beginPath();
          context.arc(
            landmark.x * canvas.width,
            landmark.y * canvas.height,
            4,
            0,
            Math.PI * 2
          );
          context.fill();
        }
      }
      sequence.push(flattenLandmarks(results));
      if (sequence.length > SEQUENCE_LENGTH) {
        sequence.shift();
      }
      progressElement.textContent = `Sequence: ${sequence.length}/${SEQUENCE_LENGTH} frame`;

      if (sequence.length === SEQUENCE_LENGTH && !predictionInFlight) {
        sendPrediction([...sequence]).catch((error) => {
          setStatus(`Gagal mengirim prediksi: ${error.message}`);
        });
      }
    }
  } catch (error) {
    setStatus(`Gagal mengambil frame: ${error.message}`);
  }
  if (running) {
    captureTimer = setTimeout(() => {
      processFrame(performance.now());
    }, 100);
  }
}

async function start() {
  if (running) {
    return;
  }
  try {
    await createLandmarker();
    sequence = [];
    lastTimestamp = -1;
    lastProcessedTime = -1;
    running = true;
    startButton.disabled = true;
    stopButton.disabled = false;
    normalizeCameraUrl(cameraBaseUrl.value);
    setStatus("Kamera aktif, mengambil snapshot...");
    processFrame(performance.now());
  } catch (error) {
    setStatus(`Gagal memulai: ${error.message}`);
  }
}

function stop() {
  running = false;
  clearTimeout(captureTimer);
  context.clearRect(0, 0, canvas.width, canvas.height);
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Berhenti");
}

startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
