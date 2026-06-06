const express = require('express');
const crypto = require('crypto');
const fetch = require('node-fetch');
const https = require('https');
const fs = require('fs');

// ---- Cấu hình ----
const API_URL = "https://treo-lc79.onrender.com";
const FETCH_TIMEOUT = 10000;        // ms
const POLL_INTERVAL = 2000;         // 2 giây
const HISTORY_WINDOW = 100;
const MAX_TRACK = 50;
const PORT = process.env.PORT || 3000;

// Vô hiệu hoá kiểm tra chứng chỉ SSL (như Python)
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

// ---- Các hàm tiện ích MD5 ----
function md5Of(text) {
  return crypto.createHash('md5').update(String(text)).digest('hex');
}

function md5NumericValue(h) {
  return parseInt(h.slice(0, 8), 16) / 0xFFFFFFFF;
}

function md5BitSum(h) {
  const ones = BigInt('0x' + h).toString(2).split('1').length - 1;
  return ones / 128.0;
}

function md5ByteMean(h) {
  let sum = 0;
  for (let i = 0; i < 32; i += 2) {
    sum += parseInt(h.substr(i, 2), 16);
  }
  return sum / (16 * 255);
}

function md5Entropy(h) {
  const freq = {};
  for (const ch of h) freq[ch] = (freq[ch] || 0) + 1;
  let ent = 0;
  const len = h.length;
  for (const c in freq) {
    const p = freq[c] / len;
    ent -= p * Math.log2(p);
  }
  return ent / 4.0;
}

function md5NibbleBias(h) {
  return [...h].filter(c => '89abcdef'.includes(c)).length / h.length;
}

function md5Xor(h) {
  return (parseInt(h[0], 16) ^ parseInt(h[31], 16)) / 15.0;
}

function analyzeMd5Layers(session) {
  const id = session._id || '';
  const phien = String(session.phien || '');
  const mi = md5Of(id);
  const mp = md5Of(phien);
  const mc = md5Of(id + phien);
  return {
    layers: {
      L1: md5NumericValue(mi),
      L2: md5BitSum(mp),
      L3: md5ByteMean(mc),
      L4: md5Entropy(mi),
      L5: md5NibbleBias(mc),
      L6: md5Xor(mi),
    },
    mi, mp, mc
  };
}

// ---- Engine thống kê lịch sử ----
class StatEngine {
  constructor(history) {
    this.results = history.map(s => s.result);
    this.points  = history.map(s => s.point);
    this.dices   = history.map(s => s.dices);
  }

  frequencyScore(w = 20) {
    const tail = this.results.slice(0, w);
    if (!tail.length) return { tai: 0.5, xiu: 0.5 };
    const cnt = { TAI: 0, XIU: 0 };
    tail.forEach(r => cnt[r]++);
    return { tai: cnt.TAI / tail.length, xiu: cnt.XIU / tail.length };
  }

  streakCurrent() {
    if (!this.results.length) return { cur: null, len: 0 };
    const cur = this.results[0];
    let len = 0;
    for (const r of this.results) {
      if (r === cur) len++;
      else break;
    }
    return { cur, len };
  }

  streakScore() {
    const { len } = this.streakCurrent();
    return Math.min(0.45 + len * 0.05, 0.85);
  }

  pointTrend(w = 15) {
    const pts = this.points.slice(0, w);
    if (!pts.length) return 0.5;
    return (pts.reduce((a, b) => a + b, 0) / pts.length - 3) / 15;
  }

  diceFaceBias(w = 20) {
    const fc = {};
    this.dices.slice(0, w).forEach(dice => {
      dice.forEach(f => fc[f] = (fc[f] || 0) + 1);
    });
    const total = Object.values(fc).reduce((a, b) => a + b, 0);
    if (!total) return 0.5;
    const high = (fc[4] || 0) + (fc[5] || 0) + (fc[6] || 0);
    return high / total;
  }

  periodScore(periods = [2, 3, 4, 5]) {
    const votes = { TAI: 0, XIU: 0 };
    for (const T of periods) {
      if (this.results.length < T * 3) continue;
      const base = this.results[0];
      let matches = 0;
      const maxIdx = Math.min(T * 4, this.results.length);
      for (let k = T; k < maxIdx; k++) {
        if (this.results[k] === base) matches++;
      }
      const total = maxIdx - T;
      if (total > 0) {
        const ratio = matches / total;
        if (ratio > 0.55) votes[base]++;
        else if (ratio < 0.45) {
          const opp = base === 'TAI' ? 'XIU' : 'TAI';
          votes[opp]++;
        }
      }
    }
    const t = votes.TAI || 0, x = votes.XIU || 0;
    return (t + x) ? t / (t + x) : 0.5;
  }

  markovScore() {
    const tr = { TAI: { TAI: 0, XIU: 0 }, XIU: { TAI: 0, XIU: 0 } };
    const tail = this.results.slice(0, HISTORY_WINDOW);
    for (let i = 0; i < tail.length - 1; i++) {
      tr[tail[i]][tail[i + 1]]++;
    }
    const cur = tail[0] || 'TAI';
    const row = tr[cur];
    const total = row.TAI + row.XIU;
    return total ? row.TAI / total : 0.5;
  }
}

// ---- Tổng hợp dự đoán ----
const W = {
  L1: 0.12, L2: 0.10, L3: 0.10, L4: 0.08, L5: 0.10, L6: 0.10,
  A: 0.10, B: 0.08, C: 0.06, D: 0.06, E: 0.05, F: 0.05
};

function aggregate(layers, stats, streakCur, streakLen) {
  const d = { ...layers };
  d.A = stats.freq_tai;
  d.C = stats.point;
  d.D = stats.dice;
  d.E = stats.period;
  d.F = stats.markov;
  const flip = stats.streak_flip;
  d.B = streakCur === 'TAI' ? (1.0 - flip) : flip;

  let score = 0;
  for (const k in W) score += W[k] * d[k];

  function sigmoid(x, c = 0.5, s = 6) {
    return 1 / (1 + Math.exp(-s * (x - c)));
  }
  const taiP = sigmoid(score);
  const xiuP = 1.0 - taiP;
  const pred = taiP > xiuP ? 'TAI' : 'XIU';
  return { pred, taiP, xiuP, detail: d, rawScore: score };
}

function consensusCheck(detail) {
  let vt = 0, vx = 0, vn = 0;
  for (const v of Object.values(detail)) {
    if (v > 0.5) vt++;
    else if (v < 0.5) vx++;
    else vn++;
  }
  const total = vt + vx + vn;
  const agree = Math.max(vt, vx) / total;
  return { vt, vx, vn, agree };
}

// ---- Trạng thái toàn cục ----
const state = {
  pendingPrediction: null,
  resolvedPredictions: [],
  lastPhien: null,
  latestData: null,
  latestRender: null,   // chuỗi text giao diện (cho console/log)
  latestPrediction: null,
  detail: null,
  rawScore: 0,
  taiP: 0, xiuP: 0,
  accuracy: { correct: 0, wrong: 0, acc: 0 },
  recentLog: []
};

// ---- Gọi API ----
async function fetchData() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT);
    const res = await fetch(API_URL, {
      headers: { 'User-Agent': 'MD5Predictor/3.0', 'Accept': 'application/json' },
      signal: controller.signal
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error(`[Lỗi mạng] ${e.message}`);
    return null;
  }
}

// ---- Vòng lặp chính (chạy nền) ----
async function mainLoop() {
  console.log("Khởi động vòng lặp dự đoán...");
  while (true) {
    const data = await fetchData();
    if (!data || !data.history || data.history.length < 5) {
      await sleep(POLL_INTERVAL);
      continue;
    }

    const history = data.history;
    const latest = history[0];
    const curPhien = latest.phien;

    // Xử lý khi có phiên mới
    if (curPhien !== state.lastPhien) {
      // Kiểm tra dự đoán cũ nếu có
      if (state.pendingPrediction) {
        state.resolvedPredictions.push({
          phien: curPhien,
          predicted: state.pendingPrediction,
          actual: latest.result
        });
        if (state.resolvedPredictions.length > MAX_TRACK * 2) {
          state.resolvedPredictions = state.resolvedPredictions.slice(-MAX_TRACK * 2);
        }
      }

      state.lastPhien = curPhien;

      // Phân tích và đưa ra dự đoán mới
      const { layers, mi, mp, mc } = analyzeMd5Layers(latest);
      const eng = new StatEngine(history);
      const { tai: freqTai } = eng.frequencyScore();
      const { cur: streakCur, len: streakLen } = eng.streakCurrent();
      const stats = {
        freq_tai: freqTai,
        point: eng.pointTrend(),
        dice: eng.diceFaceBias(),
        period: eng.periodScore(),
        markov: eng.markovScore(),
        streak_flip: eng.streakScore()
      };
      const agg = aggregate(layers, stats, streakCur, streakLen);
      state.pendingPrediction = agg.pred;
      state.taiP = agg.taiP;
      state.xiuP = agg.xiuP;
      state.detail = agg.detail;
      state.rawScore = agg.rawScore;

      const { vt, vx, vn, agree } = consensusCheck(agg.detail);

      // Lưu dữ liệu để giao diện web dùng
      state.latestData = data;
      state.latestPrediction = agg.pred;
      state.accuracy = calcAccuracy();
      state.recentLog = getRecentLog(15);

      // Tạo chuỗi giao diện (dùng cho log hoặc web)
      state.latestRender = buildRenderString(data, agg, vt, vx, vn, agree, streakCur, streakLen, mi, mp, mc);
    }

    await sleep(POLL_INTERVAL);
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function calcAccuracy() {
  const resolved = state.resolvedPredictions;
  if (!resolved.length) return { correct: 0, wrong: 0, acc: 0 };
  const correct = resolved.filter(x => x.predicted === x.actual).length;
  const wrong = resolved.length - correct;
  return { correct, wrong, acc: correct / resolved.length };
}

function getRecentLog(n) {
  return state.resolvedPredictions.slice(-n).reverse();
}

function bar(value, width = 15) {
  const filled = Math.round(value * width);
  return '#'.repeat(filled) + '-'.repeat(width - filled);
}

function pct(v) { return (v * 100).toFixed(1) + '%'; }
function arrow(v) {
  if (v > 0.52) return 'TAI^';
  if (v < 0.48) return 'XIU_';
  return ' -- ';
}

function buildRenderString(data, agg, vt, vx, vn, agree, streakCur, streakLen, mi, mp, mc) {
  const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
  const latest = data.latest;
  const phien = latest?.phien || '?';
  const dices = latest?.dices || [];
  const point = latest?.point || '?';
  const result = latest?.result || '?';
  const SEP = '='.repeat(50);
  const SEP2 = '-'.repeat(50);

  let out = '';
  out += `${SEP}\n  MD5 PREDICTION ENGINE    ${now}\n${SEP}\n`;
  out += `  Phien   : ${phien}\n  Xuc xac : [${dices.join(' ')}]   Tong: ${point}\n`;
  out += `  Ket qua : ${result}   |   Chuoi: ${streakCur} x${streakLen}\n${SEP2}\n`;

  out += '  [ MD5 LAYERS ]\n';
  const layerInfo = [
    ['L1', 'NormVal '], ['L2', 'BitSum  '], ['L3', 'ByteMean'],
    ['L4', 'Entropy '], ['L5', 'Nibble  '], ['L6', 'XorChk  ']
  ];
  for (const [k, lbl] of layerInfo) {
    const v = agg.detail[k];
    out += `  ${k}:${lbl} [${bar(v, 12)}] ${pct(v).padStart(6)}  ${arrow(v)}\n`;
  }

  out += `${SEP2}\n  [ THONG KE LICH SU ]\n`;
  const statInfo = [
    ['A', 'Tan suat '], ['B', 'Dao chieu'], ['C', 'Xu diem  '],
    ['D', 'Xuc xac  '], ['E', 'Chu ky   '], ['F', 'Markov   ']
  ];
  for (const [k, lbl] of statInfo) {
    const v = agg.detail[k];
    const extra = k === 'B' ? ` (streak ${streakCur} x${streakLen})` : '';
    out += `  ${k}:${lbl} [${bar(v, 12)}] ${pct(v).padStart(6)}  ${arrow(v)}${extra}\n`;
  }

  out += `${SEP2}\n  Phieu bau:  TAI=${vt}  XIU=${vx}  Trung=${vn}   Dong thuan: ${(agree*100).toFixed(0)}%\n${SEP}\n`;

  const margin = Math.abs(agg.taiP - agg.xiuP);
  const conf = Math.min(1.0, agree * 0.5 + margin);
  const confLabel = conf >= 0.65 ? 'CAO' : (conf >= 0.45 ? 'TRUNG BINH' : 'THAP');

  out += `  >> DU DOAN PHIEN TIEP: ${agg.pred} <<\n`;
  out += `  TAI [${bar(agg.taiP, 15)}] ${pct(agg.taiP)}\n`;
  out += `  XIU [${bar(agg.xiuP, 15)}] ${pct(agg.xiuP)}\n`;
  out += `  Raw: ${agg.rawScore.toFixed(4)}   Tin cay: ${confLabel} (${pct(conf)})\n${SEP}\n`;

  const acc = calcAccuracy();
  out += '  [ THONG KE DU DOAN ]\n';
  if (acc.correct + acc.wrong > 0) {
    out += `  Tong:${acc.correct+acc.wrong}  Dung:${acc.correct}  Sai:${acc.wrong}   Chinh xac: ${pct(acc.acc)}\n`;
    out += `  [${bar(acc.acc, 20)}]\n`;
  } else {
    out += '  Chua co du lieu (can >= 1 phien da kiem chung)\n';
  }

  out += `${SEP2}\n  ${'PHIEN'.padEnd(10)} ${'THUC TE'.padEnd(8)} ${'DU DOAN'.padEnd(8)} DANH GIA\n${SEP2}\n`;
  const recent = getRecentLog(15);
  if (recent.length) {
    for (const e of recent) {
      const ok = e.predicted === e.actual ? 'DUNG [v]' : 'SAI  [x]';
      out += `  ${String(e.phien).padEnd(10)} ${e.actual.padEnd(8)} ${e.predicted.padEnd(8)} ${ok}\n`;
    }
  } else {
    out += '  (Dang cho phien dau tien hoan tat...)\n';
  }

  out += `${SEP}\n  20 phien: `;
  const hist20 = data.history.slice(0, 20);
  out += hist20.map(s => s.result[0] + s.point).join(' ') + '\n';
  out += `  [Kiem tra moi ${POLL_INTERVAL/1000}s | Ctrl+C thoat]\n${SEP}`;
  return out;
}

// ---- Máy chủ Express ----
const app = express();
app.use(express.static('public')); // (tùy chọn) nếu muốn phục vụ file tĩnh

// Trang chủ: hiển thị giao diện giống terminal
app.get('/', (req, res) => {
  const render = state.latestRender || 'Đang chờ dữ liệu...';
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>MD5 Predictor</title>
      <style>
        body { background: #000; color: #0f0; font-family: monospace; white-space: pre; margin: 10px; }
        a { color: #0f0; }
      </style>
      <meta http-equiv="refresh" content="5">
    </head>
    <body>${render.replace(/\n/g, '<br>').replace(/ /g, '&nbsp;')}</body>
    </html>
  `);
});

// API JSON cho client khác
app.get('/api', (req, res) => {
  res.json({
    latestPrediction: state.latestPrediction,
    pendingPrediction: state.pendingPrediction,
    taiP: state.taiP,
    xiuP: state.xiuP,
    detail: state.detail,
    rawScore: state.rawScore,
    accuracy: state.accuracy,
    recentLog: state.recentLog,
    lastPhien: state.lastPhien
  });
});

// Khởi động server và vòng lặp
app.listen(PORT, () => {
  console.log(`Server đang chạy tại http://localhost:${PORT}`);
  mainLoop().catch(err => console.error('Lỗi vòng lặp:', err));
});