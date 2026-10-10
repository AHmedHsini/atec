const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const colors = { rock: "#e07a4c", crater: "#c8b89a", ice: "#9fd7ff" };

function clock(seconds) {
  const t = Math.max(0, seconds);
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `T+${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function draw(state) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0b0d12";
  ctx.fillRect(0, 0, w, h);

  const originX = w * 0.48;
  const originY = h * 0.78;
  const scale = 28;

  ctx.strokeStyle = "#243044";
  ctx.lineWidth = 1;
  for (let m = -8; m <= 18; m += 2) {
    const y = originY - m * scale;
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#f0b429";
  ctx.beginPath();
  ctx.arc(originX, originY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#93a0b8";
  ctx.font = "12px Segoe UI";
  ctx.fillText("Site Alpha", originX + 8, originY + 4);

  const rover = state.rover;
  const rx = originX + rover.y * scale;
  const ry = originY - rover.x * scale;
  ctx.save();
  ctx.translate(rx, ry);
  ctx.rotate((rover.yaw * Math.PI) / 180);
  ctx.fillStyle = state.blackout ? "#ff5d5d" : "#3ddc97";
  ctx.beginPath();
  ctx.moveTo(0, -10);
  ctx.lineTo(7, 8);
  ctx.lineTo(-7, 8);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  for (const track of state.tracks) {
    const x = originX + track.y * scale;
    const y = originY - track.x * scale;
    const radius = Math.max(8, track.uncertainty_m * scale);
    ctx.beginPath();
    ctx.strokeStyle = colors[track.kind] || "#fff";
    ctx.globalAlpha = Math.max(0.25, 1 - track.age_s / 8);
    ctx.lineWidth = 2;
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = colors[track.kind] || "#fff";
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e7ecf5";
    ctx.fillText(`${track.kind} ${track.id} · ${track.age_s.toFixed(1)}s`, x + 8, y - 8);
  }
}

async function tick() {
  const response = await fetch("/api/state");
  const state = await response.json();
  document.getElementById("clock").textContent = clock(state.time_s);
  document.getElementById("chapter").textContent = state.chapter;
  const survival = Math.round(state.link.survival * 100);
  document.getElementById("link").textContent = state.blackout ? "BLACKOUT" : `${survival}% in`;
  document.getElementById("bytes").textContent = `${state.link.packet_bytes} B vs ${state.link.json_bytes} B`;
  document.getElementById("fps").textContent = `${state.perception.fps} FPS`;
  document.getElementById("error").textContent = `${state.perception.localization_error_m} m`;
  document.getElementById("banner").classList.toggle("show", state.blackout);
  const list = document.getElementById("story");
  list.innerHTML = "";
  for (const line of [...state.story].reverse()) {
    const item = document.createElement("li");
    const stamp = document.createElement("time");
    stamp.textContent = `${clock(line.time_s)}  ${line.chapter}`;
    item.append(stamp, document.createTextNode(line.text));
    list.append(item);
  }
  draw(state);
}

setInterval(() => {
  tick().catch(() => {});
}, 400);
tick().catch(() => {});
