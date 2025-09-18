const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const serverUrlInput = document.getElementById('serverUrl');

document.getElementById('year').textContent = new Date().getFullYear();

let sending = false;
let timer = null;
let sessionId = self.crypto ? crypto.randomUUID() : String(Math.random());

const countBlinks = document.getElementById('countBlinks');
const countBrows = document.getElementById('countBrows');
const countMouth = document.getElementById('countMouth');
const stateEye = document.getElementById('stateEye');
const stateBrow = document.getElementById('stateBrow');
const stateMouth = document.getElementById('stateMouth');

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  video.srcObject = stream;
  await video.play();
  overlay.width  = video.videoWidth;
  overlay.height = video.videoHeight;
}

function snapshot() {
  // Dibuja el frame del video SOLO para codificar y enviar
  ctx.drawImage(video, 0, 0, overlay.width, overlay.height);
  const dataUrl = overlay.toDataURL('image/jpeg', 0.85);
  // Limpia inmediatamente para que el canvas quede transparente visualmente
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  return dataUrl;
}

async function sendFrame() {
  if (!sending) return;
  const b64 = snapshot();
  try {
    const url = `${serverUrlInput.value.replace(/\/$/, '')}/process`; // <-- fixed backtick
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, image_b64: b64 })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Actualiza contadores y estados
    countBlinks.textContent = data.counts.blinks;
    countBrows.textContent  = data.counts.brow_raises;
    countMouth.textContent  = data.counts.mouth_opens;

    stateEye.textContent   = `Ojos: ${data.states.eye}`;
    stateBrow.textContent  = `Cejas: ${data.states.brow}`;
    stateMouth.textContent = `Boca: ${data.states.mouth}`;

    // Dibuja SOLO landmarks (no repintes el video)
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (data.landmarks && data.landmarks.length) {
      ctx.beginPath();
      for (const p of data.landmarks) {
        const x = p.x * overlay.width;
        const y = p.y * overlay.height;
        ctx.moveTo(x + 1.2, y);
        ctx.arc(x, y, 1.2, 0, Math.PI * 2);
      }
      ctx.fillStyle = 'rgba(0, 123, 255, 0.9)';
      ctx.fill();
    }
  } catch (err) {
    console.error('Error enviando frame:', err);
  }
}

btnStart.addEventListener('click', async () => {
  console.log('Iniciar clic'); // debug
  if (!video.srcObject) await setupCamera();
  sending = true;
  if (timer) clearInterval(timer);
  timer = setInterval(sendFrame, 200); // estabilidad
});

btnStop.addEventListener('click', () => {
  sending = false;
  if (timer) clearInterval(timer);
});

window.addEventListener('DOMContentLoaded', () => {});
