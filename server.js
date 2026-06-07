const express = require('express');
const crypto = require('crypto');
const fetch = require('node-fetch');
const fs = require('fs');

// ---- Cấu hình ----
const API_URL = "https://treo-lc79.onrender.com";
const FETCH_TIMEOUT = 10000;
const POLL_INTERVAL = 2000;
const HISTORY_WINDOW = 100;
const MAX_TRACK = 200;        // Tăng lên 200 để học sâu hơn
const PORT = process.env.PORT || 3000;
const WEIGHT_SAVE_FILE = './adaptive_weights.json';

process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

// ============================================================
//  ADAPTIVE WEIGHT SYSTEM — Trọng số học từ lịch sử
// ============================================================
const DEFAULT_WEIGHTS = {
  L1: 0.12, L2: 0.10, L3: 0.10, L4: 0.08, L5: 0.10, L6: 0.10,
  A: 0.10, B: 0.08, C: 0.06, D: 0.06, E: 0.05, F: 0.05
};

class AdaptiveWeightManager {
  constructor() {
    this.weights = { ...DEFAULT_WEIGHTS };
    this.featureHistory = {}; // { featureName: [{value, correct}] }
    this.featureAccuracy = {};
    this.updateCount = 0;
    this.load();
  }

  load() {
    try {
      if (fs.existsSync(WEIGHT_SAVE_FILE)) {
        const data = JSON.parse(fs.readFileSync(WEIGHT_SAVE_FILE, 'utf8'));
        this.weights = data.weights || { ...DEFAULT_WEIGHTS };
        this.featureHistory = data.featureHistory || {};
        this.featureAccuracy = data.featureAccuracy || {};
        this.updateCount = data.updateCount || 0;
        console.log(`[AdaptiveWeights] Đã tải ${this.updateCount} lần cập nhật từ file.`);
      }
    } catch (e) {
      console.log('[AdaptiveWeights] Bắt đầu với trọng số mặc định.');
    }
  }

  save() {
    try {
      fs.writeFileSync(WEIGHT_SAVE_FILE, JSON.stringify({
        weights: this.weights,
        featureHistory: this.featureHistory,
        featureAccuracy: this.featureAccuracy,
        updateCount: this.updateCount
      }, null, 2));
    } catch (e) {}
  }

  // Ghi nhận kết quả của một phiên dự đoán
  recordOutcome(detail, prediction, actual) {
    const correct = prediction === actual;
    const actualBias = actual === 'TAI' ? 1.0 : 0.0;

    for (const [key, value] of Object.entries(detail)) {
      if (!this.featureHistory[key]) this.featureHistory[key] = [];
      // Tính xem feature này có "đúng hướng" không
      const featureCorrect = (value > 0.5 && actual === 'TAI') || (value < 0.5 && actual === 'XIU');
      this.featureHistory[key].push({ value, correct: featureCorrect, actualBias });
      // Giữ tối đa 500 mẫu
      if (this.featureHistory[key].length > 500) {
        this.featureHistory[key] = this.featureHistory[key].slice(-500);
      }
    }

    this.updateCount++;
    // Cập nhật trọng số mỗi 5 phiên
    if (this.updateCount % 5 === 0) {
      this.rebalanceWeights();
      this.save();
    }
  }

  rebalanceWeights() {
    const accuracies = {};
    let totalAcc = 0;
    let count = 0;

    for (const [key, history] of Object.entries(this.featureHistory)) {
      if (history.length < 10) continue;
      const correct = history.filter(h => h.correct).length;
      const acc = correct / history.length;
      accuracies[key] = acc;
      this.featureAccuracy[key] = acc;
      totalAcc += acc;
      count++;
    }

    if (count === 0) return;

    // Điều chỉnh trọng số dựa trên accuracy (softmax-style)
    const avgAcc = totalAcc / count;
    const newWeights = {};
    let totalWeight = 0;

    for (const key of Object.keys(DEFAULT_WEIGHTS)) {
      const acc = accuracies[key] || 0.5;
      // Feature chính xác hơn trung bình → tăng trọng số
      // Dùng exponential scaling để tạo sự phân biệt rõ ràng
      const factor = Math.exp((acc - 0.5) * 4);
      newWeights[key] = DEFAULT_WEIGHTS[key] * factor;
      totalWeight += newWeights[key];
    }

    // Chuẩn hoá để tổng = 1
    for (const key of Object.keys(newWeights)) {
      this.weights[key] = newWeights[key] / totalWeight;
    }

    console.log(`[AdaptiveWeights] Cập nhật trọng số lần #${this.updateCount}`);
  }

  getWeights() { return this.weights; }
  getAccuracies() { return this.featureAccuracy; }
}

// ============================================================
//  CASINO MANIPULATION DETECTOR — Phát hiện nhà cái can thiệp
// ============================================================
class ManipulationDetector {
  constructor() {
    this.windowSize = 30;
    this.alerts = [];
    this.manipulationScore = 0;
    this.isManipulated = false;
    this.manipulationType = null;
  }

  analyze(history) {
    if (history.length < 10) return this.buildReport(false, null, 0);

    const results = history.map(h => h.result).slice(0, this.windowSize);
    const points = history.map(h => h.point).slice(0, this.windowSize);
    const dices = history.map(h => h.dices).slice(0, this.windowSize);

    const signals = [];

    // === Kiểm tra 1: Mất cân bằng tần suất bất thường ===
    const taiCount = results.filter(r => r === 'TAI').length;
    const ratio = taiCount / results.length;
    if (ratio > 0.75 || ratio < 0.25) {
      signals.push({
        type: 'FREQ_BIAS',
        severity: 'HIGH',
        desc: `Tỷ lệ TAI/XIU lệch nặng: ${(ratio*100).toFixed(1)}%`,
        score: Math.abs(ratio - 0.5) * 2
      });
    }

    // === Kiểm tra 2: Cầu dài bất thường (>6) ===
    let maxStreak = 0, curStreak = 1;
    for (let i = 1; i < results.length; i++) {
      if (results[i] === results[i-1]) curStreak++;
      else { maxStreak = Math.max(maxStreak, curStreak); curStreak = 1; }
    }
    maxStreak = Math.max(maxStreak, curStreak);
    if (maxStreak >= 7) {
      signals.push({
        type: 'LONG_STREAK',
        severity: 'MEDIUM',
        desc: `Cầu dài bất thường: ${maxStreak} phiên liên tiếp`,
        score: Math.min((maxStreak - 5) * 0.15, 0.6)
      });
    }

    // === Kiểm tra 3: Phá cầu đúng thời điểm người chơi đặt nhiều nhất ===
    // Nhận diện pattern: đang cầu dài → đột ngột đảo → cầu dài bên kia
    const flipPattern = this.detectFlipPattern(results);
    if (flipPattern.detected) {
      signals.push({
        type: 'FLIP_PATTERN',
        severity: 'HIGH',
        desc: `Pattern đảo cầu có chủ đích: ${flipPattern.desc}`,
        score: 0.7
      });
    }

    // === Kiểm tra 4: Xúc xắc bất thường (phân phối điểm) ===
    const diceAnomaly = this.analyzeDiceDistribution(points);
    if (diceAnomaly.anomaly) {
      signals.push({
        type: 'DICE_ANOMALY',
        severity: 'MEDIUM',
        desc: diceAnomaly.desc,
        score: diceAnomaly.score
      });
    }

    // === Kiểm tra 5: Phá dự đoán chính xác quá thường xuyên ===
    // Nếu model đang đúng nhiều → nhà cái "cảm nhận" và đảo
    const manipScore = signals.reduce((acc, s) => acc + s.score, 0) / Math.max(signals.length, 1);
    const totalScore = Math.min(manipScore * (signals.length > 0 ? 1.2 : 1), 1.0);

    this.manipulationScore = totalScore;
    this.isManipulated = totalScore > 0.45;
    this.manipulationType = signals.length > 0 ? signals[0].type : null;
    this.alerts = signals;

    return this.buildReport(this.isManipulated, signals, totalScore);
  }

  detectFlipPattern(results) {
    // Tìm pattern: A*n → B*m → A*n (n,m >= 3)
    let i = 0;
    while (i < results.length - 6) {
      const base = results[i];
      let runA = 0;
      while (i + runA < results.length && results[i + runA] === base) runA++;
      if (runA >= 3) {
        const opp = base === 'TAI' ? 'XIU' : 'TAI';
        let runB = 0;
        while (i + runA + runB < results.length && results[i + runA + runB] === opp) runB++;
        if (runB >= 3) {
          return { detected: true, desc: `${base}x${runA} → ${opp}x${runB}` };
        }
      }
      i += Math.max(runA, 1);
    }
    return { detected: false };
  }

  analyzeDiceDistribution(points) {
    if (points.length < 15) return { anomaly: false };
    const avg = points.reduce((a, b) => a + b, 0) / points.length;
    // Điểm trung bình lý thuyết = 10.5, std ≈ 2.96
    const deviation = Math.abs(avg - 10.5);
    if (deviation > 2.5) {
      return {
        anomaly: true,
        desc: `Điểm xúc xắc trung bình lệch: ${avg.toFixed(1)} (chuẩn: 10.5)`,
        score: Math.min(deviation / 5, 0.8)
      };
    }
    return { anomaly: false };
  }

  buildReport(isManipulated, signals, score) {
    return { isManipulated, signals: signals || [], score, type: this.manipulationType };
  }

  // Gợi ý chiến lược khi phát hiện can thiệp
  getCounterStrategy(currentPrediction, manipReport) {
    if (!manipReport.isManipulated) return currentPrediction;

    const score = manipReport.score;
    const signals = manipReport.signals.map(s => s.type);

    // Nếu đang có flip pattern → bẻ ngược dự đoán
    if (signals.includes('FLIP_PATTERN') && score > 0.6) {
      return currentPrediction === 'TAI' ? 'XIU' : 'TAI';
    }

    // Nếu tần suất lệch → đặt ngược chiều lệch
    if (signals.includes('FREQ_BIAS')) {
      const freqSignal = manipReport.signals.find(s => s.type === 'FREQ_BIAS');
      if (freqSignal && score > 0.55) {
        return currentPrediction === 'TAI' ? 'XIU' : 'TAI';
      }
    }

    return currentPrediction;
  }
}

// ============================================================
//  PATTERN MEMORY — Ghi nhớ và học từ các mẫu cầu
// ============================================================
class PatternMemory {
  constructor() {
    this.patterns = {}; // { "patternKey": { tai: n, xiu: n } }
    this.PATTERN_LENGTHS = [3, 4, 5, 6];
  }

  // Mã hoá chuỗi thành key
  encodePattern(results, len) {
    return results.slice(0, len).map(r => r[0]).join('');
  }

  // Học từ lịch sử
  train(history) {
    for (const len of this.PATTERN_LENGTHS) {
      for (let i = 0; i < history.length - len; i++) {
        const pattern = this.encodePattern(history.slice(i + 1).map(h => h.result), len);
        const next = history[i].result;
        if (!this.patterns[pattern]) this.patterns[pattern] = { TAI: 0, XIU: 0, total: 0 };
        this.patterns[pattern][next]++;
        this.patterns[pattern].total++;
      }
    }
  }

  // Dự đoán dựa trên pattern hiện tại
  predict(recentResults) {
    let bestConfidence = 0.5;
    let bestPrediction = null;
    let matchedPattern = null;
    let matchedStats = null;

    for (const len of this.PATTERN_LENGTHS.slice().reverse()) { // ưu tiên pattern dài
      if (recentResults.length < len) continue;
      const pattern = this.encodePattern(recentResults, len);
      const stats = this.patterns[pattern];
      if (!stats || stats.total < 5) continue; // cần ít nhất 5 mẫu

      const taiRate = stats.TAI / stats.total;
      const confidence = Math.abs(taiRate - 0.5) * 2; // 0..1

      if (confidence > bestConfidence) {
        bestConfidence = confidence;
        bestPrediction = taiRate > 0.5 ? 'TAI' : 'XIU';
        matchedPattern = pattern;
        matchedStats = { ...stats, taiRate };
      }
    }

    return {
      prediction: bestPrediction,
      confidence: bestConfidence,
      pattern: matchedPattern,
      stats: matchedStats
    };
  }

  getTotalPatterns() {
    return Object.keys(this.patterns).length;
  }
}

// ============================================================
//  CÁC HÀM MD5 (giữ nguyên)
// ============================================================
function md5Of(text) {
  return crypto.createHash('md5').update(String(text)).digest('hex');
}
function md5NumericValue(h) { return parseInt(h.slice(0, 8), 16) / 0xFFFFFFFF; }
function md5BitSum(h) {
  const ones = BigInt('0x' + h).toString(2).split('1').length - 1;
  return ones / 128.0;
}
function md5ByteMean(h) {
  let sum = 0;
  for (let i = 0; i < 32; i += 2) sum += parseInt(h.substr(i, 2), 16);
  return sum / (16 * 255);
}
function md5Entropy(h) {
  const freq = {};
  for (const ch of h) freq[ch] = (freq[ch] || 0) + 1;
  let ent = 0;
  const len = h.length;
  for (const c in freq) { const p = freq[c] / len; ent -= p * Math.log2(p); }
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
  const mi = md5Of(id), mp = md5Of(phien), mc = md5Of(id + phien);
  return {
    layers: {
      L1: md5NumericValue(mi), L2: md5BitSum(mp), L3: md5ByteMean(mc),
      L4: md5Entropy(mi), L5: md5NibbleBias(mc), L6: md5Xor(mi)
    },
    mi, mp, mc
  };
}

// ============================================================
//  STAT ENGINE (nâng cấp)
// ============================================================
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
    for (const r of this.results) { if (r === cur) len++; else break; }
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
    for (let i = 0; i < tail.length - 1; i++) tr[tail[i]][tail[i + 1]]++;
    const cur = tail[0] || 'TAI';
    const row = tr[cur];
    const total = row.TAI + row.XIU;
    return total ? row.TAI / total : 0.5;
  }

  // MỚI: Markov bậc 2 (xem xét 2 phiên trước)
  markov2Score() {
    const tail = this.results.slice(0, HISTORY_WINDOW);
    if (tail.length < 3) return 0.5;
    const tr = {};
    for (let i = 0; i < tail.length - 2; i++) {
      const key = `${tail[i+1]}_${tail[i]}`;
      if (!tr[key]) tr[key] = { TAI: 0, XIU: 0 };
      tr[key][tail[i + 2 - 2]]++;  // sửa: dự đoán [i]
    }
    // Thực chất: cho 2 kết quả gần nhất, kết quả tiếp theo là gì
    const markov2 = {};
    for (let i = 0; i < tail.length - 2; i++) {
      const key = `${tail[i]}_${tail[i+1]}`;
      if (!markov2[key]) markov2[key] = { TAI: 0, XIU: 0 };
      markov2[key][tail[i+2]]++;
    }
    if (tail.length < 2) return 0.5;
    const curKey = `${tail[0]}_${tail[1]}`;
    const stats = markov2[curKey];
    if (!stats) return 0.5;
    const total = (stats.TAI || 0) + (stats.XIU || 0);
    return total ? (stats.TAI || 0) / total : 0.5;
  }

  // MỚI: Phát hiện chu kỳ bằng autocorrelation
  autocorrelation(lag = 5) {
    const binary = this.results.slice(0, 50).map(r => r === 'TAI' ? 1 : 0);
    if (binary.length < lag + 5) return 0.5;
    let match = 0;
    for (let i = 0; i < binary.length - lag; i++) {
      if (binary[i] === binary[i + lag]) match++;
    }
    return match / (binary.length - lag);
  }
}

// ============================================================
//  AGGREGATE — Dự đoán tổng hợp (dùng adaptive weights)
// ============================================================
function aggregate(layers, stats, streakCur, streakLen, weights, patternPred) {
  const d = { ...layers };
  d.A = stats.freq_tai;
  d.C = stats.point;
  d.D = stats.dice;
  d.E = stats.period;
  d.F = stats.markov;
  const flip = stats.streak_flip;
  d.B = streakCur === 'TAI' ? (1.0 - flip) : flip;

  let score = 0;
  let totalW = 0;
  for (const k in weights) {
    if (d[k] !== undefined) {
      score += weights[k] * d[k];
      totalW += weights[k];
    }
  }
  if (totalW > 0) score /= totalW;

  // Tích hợp dự đoán Pattern Memory (nếu có, có trọng số 0.2)
  let patternBoost = 0;
  if (patternPred && patternPred.prediction && patternPred.confidence > 0.15) {
    const patternScore = patternPred.prediction === 'TAI' ? patternPred.confidence : 1 - patternPred.confidence;
    score = score * 0.82 + patternScore * 0.18;
    patternBoost = patternPred.confidence;
  }

  // Markov bậc 2
  if (stats.markov2 !== undefined) {
    score = score * 0.92 + stats.markov2 * 0.08;
  }

  function sigmoid(x, c = 0.5, s = 8) {
    return 1 / (1 + Math.exp(-s * (x - c)));
  }

  const taiP = sigmoid(score);
  const xiuP = 1.0 - taiP;
  const pred = taiP > xiuP ? 'TAI' : 'XIU';
  return { pred, taiP, xiuP, detail: d, rawScore: score, patternBoost };
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

// ============================================================
//  GLOBAL STATE
// ============================================================
const weightManager = new AdaptiveWeightManager();
const manipDetector = new ManipulationDetector();
const patternMemory = new PatternMemory();

const state = {
  pendingPrediction: null,
  pendingDetail: null,
  resolvedPredictions: [],
  lastPhien: null,
  latestData: null,
  latestRender: null,
  latestPrediction: null,
  finalPrediction: null,      // Sau khi điều chỉnh manipulation
  detail: null,
  rawScore: 0,
  taiP: 0, xiuP: 0,
  accuracy: { correct: 0, wrong: 0, acc: 0 },
  recentLog: [],
  manipReport: null,
  patternPred: null,
  adaptiveLearningStats: { totalUpdates: 0, featureAccuracies: {} },
  learningPhase: 'COLLECTING', // COLLECTING, LEARNING, OPTIMIZED
  streakInfo: { cur: null, len: 0 }
};

// ============================================================
//  API + MAIN LOOP
// ============================================================
async function fetchData() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT);
    const res = await fetch(API_URL, {
      headers: { 'User-Agent': 'MD5PredictorV4/adaptive', 'Accept': 'application/json' },
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

async function mainLoop() {
  console.log("Khởi động MD5 Predictor V4 — Adaptive Learning Engine...");
  while (true) {
    const data = await fetchData();
    if (!data || !data.history || data.history.length < 5) {
      await sleep(POLL_INTERVAL);
      continue;
    }

    const history = data.history;
    const latest = history[0];
    const curPhien = latest.phien;

    if (curPhien !== state.lastPhien) {
      // === Bước 1: Xử lý kết quả phiên cũ (học) ===
      if (state.pendingPrediction && state.pendingDetail) {
        const actual = latest.result;
        const predicted = state.pendingPrediction;

        // Ghi kết quả để học
        weightManager.recordOutcome(state.pendingDetail, predicted, actual);

        // Lưu vào log
        state.resolvedPredictions.push({
          phien: curPhien,
          predicted,
          actual,
          finalPredicted: state.finalPrediction || predicted,
          correct: (state.finalPrediction || predicted) === actual,
          manipDetected: state.manipReport?.isManipulated || false
        });
        if (state.resolvedPredictions.length > MAX_TRACK * 2) {
          state.resolvedPredictions = state.resolvedPredictions.slice(-MAX_TRACK * 2);
        }
      }

      state.lastPhien = curPhien;

      // === Bước 2: Huấn luyện Pattern Memory ===
      patternMemory.train(history.slice(0, 50));

      // === Bước 3: Phân tích manipulation ===
      const manipReport = manipDetector.analyze(history);
      state.manipReport = manipReport;

      // === Bước 4: Dự đoán Pattern ===
      const recentResults = history.slice(0, 10).map(h => h.result);
      const patternPred = patternMemory.predict(recentResults);
      state.patternPred = patternPred;

      // === Bước 5: Phân tích MD5 + Thống kê ===
      const { layers, mi, mp, mc } = analyzeMd5Layers(latest);
      const eng = new StatEngine(history);
      const { tai: freqTai } = eng.frequencyScore();
      const { cur: streakCur, len: streakLen } = eng.streakCurrent();
      state.streakInfo = { cur: streakCur, len: streakLen };

      const stats = {
        freq_tai: freqTai,
        point: eng.pointTrend(),
        dice: eng.diceFaceBias(),
        period: eng.periodScore(),
        markov: eng.markovScore(),
        markov2: eng.markov2Score(),
        streak_flip: eng.streakScore()
      };

      // === Bước 6: Aggregate với adaptive weights ===
      const weights = weightManager.getWeights();
      const agg = aggregate(layers, stats, streakCur, streakLen, weights, patternPred);

      // === Bước 7: Điều chỉnh theo manipulation ===
      const counterPred = manipDetector.getCounterStrategy(agg.pred, manipReport);

      state.pendingPrediction = agg.pred;
      state.pendingDetail = agg.detail;
      state.finalPrediction = counterPred;
      state.taiP = counterPred === 'TAI' ? agg.taiP : agg.xiuP;
      state.xiuP = counterPred === 'XIU' ? agg.taiP : agg.xiuP;
      state.detail = agg.detail;
      state.rawScore = agg.rawScore;

      // === Bước 8: Cập nhật meta-state ===
      state.latestData = data;
      state.latestPrediction = counterPred;
      state.accuracy = calcAccuracy();
      state.recentLog = getRecentLog(15);
      state.adaptiveLearningStats = {
        totalUpdates: weightManager.updateCount,
        featureAccuracies: weightManager.getAccuracies()
      };

      // Learning phase
      const total = state.resolvedPredictions.length;
      if (total < 20) state.learningPhase = 'COLLECTING';
      else if (total < 80) state.learningPhase = 'LEARNING';
      else state.learningPhase = 'OPTIMIZED';

      const { vt, vx, vn, agree } = consensusCheck(agg.detail);
      state.latestRender = buildRenderString(
        data, agg, vt, vx, vn, agree, streakCur, streakLen,
        mi, mp, mc, manipReport, counterPred, patternPred, weights
      );
    }

    await sleep(POLL_INTERVAL);
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function calcAccuracy() {
  const resolved = state.resolvedPredictions;
  if (!resolved.length) return { correct: 0, wrong: 0, acc: 0 };
  const correct = resolved.filter(x => x.correct).length;
  const wrong = resolved.length - correct;
  return { correct, wrong, acc: correct / resolved.length };
}

function getRecentLog(n) {
  return state.resolvedPredictions.slice(-n).reverse();
}

// ============================================================
//  RENDER — Giao diện terminal
// ============================================================
function bar(value, width = 15) {
  const filled = Math.round(value * width);
  return '#'.repeat(filled) + '-'.repeat(width - filled);
}
function pct(v) { return (v * 100).toFixed(1) + '%'; }
function arrow(v) {
  if (v > 0.55) return 'TAI^';
  if (v < 0.45) return 'XIU_';
  return ' -- ';
}

function buildRenderString(data, agg, vt, vx, vn, agree, streakCur, streakLen, mi, mp, mc, manipReport, finalPred, patternPred, weights) {
  const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
  const latest = data.latest || data.history[0];
  const phien = latest?.phien || '?';
  const dices = latest?.dices || [];
  const point = latest?.point || '?';
  const result = latest?.result || '?';
  const SEP = '='.repeat(56);
  const SEP2 = '-'.repeat(56);

  let out = '';
  out += `${SEP}\n  MD5 PREDICTOR V4 — ADAPTIVE LEARNING    ${now}\n${SEP}\n`;

  // Phase indicator
  const phase = state.learningPhase;
  const phaseLabel = phase === 'COLLECTING' ? '[ ĐANG THU THẬP DỮ LIỆU ]'
    : phase === 'LEARNING' ? '[ ĐANG HỌC - TRỌNG SỐ TỰ ĐIỀU CHỈNH ]'
    : '[ TỐI ƯU HOÁ - ĐẦY ĐỦ KHẢ NĂNG ]';
  out += `  ${phaseLabel}\n${SEP2}\n`;

  out += `  Phiên   : ${phien}\n`;
  out += `  Xúc xắc : [${dices.join(' ')}]   Tổng: ${point}\n`;
  out += `  Kết quả : ${result}   |   Cầu: ${streakCur} x${streakLen}\n${SEP2}\n`;

  // === CẢNH BÁO MANIPULATION ===
  if (manipReport && manipReport.isManipulated) {
    out += `  !!!  CẢNH BÁO: PHÁT HIỆN NHÀ CÁI CAN THIỆP  !!!\n`;
    out += `  Điểm nghi ngờ: ${pct(manipReport.score)}   Loại: ${manipReport.type || 'UNKNOWN'}\n`;
    for (const sig of manipReport.signals) {
      out += `  >> [${sig.severity}] ${sig.desc}\n`;
    }
    if (finalPred !== agg.pred) {
      out += `  ** BẺ HƯỚNG: ${agg.pred} → ${finalPred} **\n`;
    }
    out += `${SEP2}\n`;
  }

  // === MD5 LAYERS ===
  out += '  [ MD5 LAYERS ]\n';
  const layerInfo = [
    ['L1','NormVal '], ['L2','BitSum  '], ['L3','ByteMean'],
    ['L4','Entropy '], ['L5','Nibble  '], ['L6','XorChk  ']
  ];
  for (const [k, lbl] of layerInfo) {
    const v = agg.detail[k];
    const w = weights[k] || 0;
    out += `  ${k}:${lbl} [${bar(v,12)}] ${pct(v).padStart(6)}  ${arrow(v)}  w=${(w*100).toFixed(1)}%\n`;
  }

  // === THỐNG KÊ LỊCH SỬ ===
  out += `${SEP2}\n  [ THỐNG KÊ LỊCH SỬ ]\n`;
  const statInfo = [
    ['A','Tần suất '], ['B','Đảo chiều'], ['C','Xu điểm  '],
    ['D','Xúc xắc '], ['E','Chu kỳ  '], ['F','Markov   ']
  ];
  for (const [k, lbl] of statInfo) {
    const v = agg.detail[k];
    const w = weights[k] || 0;
    const acc = state.adaptiveLearningStats.featureAccuracies[k];
    const accStr = acc !== undefined ? ` acc=${pct(acc)}` : '';
    out += `  ${k}:${lbl} [${bar(v,12)}] ${pct(v).padStart(6)}  ${arrow(v)}  w=${(w*100).toFixed(1)}%${accStr}\n`;
  }

  // === PATTERN MEMORY ===
  out += `${SEP2}\n  [ PATTERN MEMORY — ${patternMemory.getTotalPatterns()} mẫu ]\n`;
  if (patternPred && patternPred.prediction) {
    out += `  Mẫu khớp : "${patternPred.pattern}"  → Dự đoán: ${patternPred.prediction}`;
    out += `  (tin cậy: ${pct(patternPred.confidence)})\n`;
    if (patternPred.stats) {
      out += `  Lịch sử mẫu: TAI=${patternPred.stats.TAI} / XIU=${patternPred.stats.XIU} (n=${patternPred.stats.total})\n`;
    }
  } else {
    out += `  Chưa đủ dữ liệu pattern (cần ít nhất 5 lần xuất hiện)\n`;
  }

  // === ADAPTIVE LEARNING ===
  out += `${SEP2}\n  [ ADAPTIVE LEARNING — Cập nhật #${state.adaptiveLearningStats.totalUpdates} ]\n`;
  const accs = state.adaptiveLearningStats.featureAccuracies;
  if (Object.keys(accs).length > 0) {
    const entries = Object.entries(accs).sort((a,b) => b[1]-a[1]);
    out += `  Top feature hiệu quả:\n`;
    for (const [k, v] of entries.slice(0, 4)) {
      const bar10 = '#'.repeat(Math.round(v*10)) + '-'.repeat(10 - Math.round(v*10));
      out += `  ${k.padEnd(3)}: [${bar10}] ${pct(v)}\n`;
    }
  } else {
    out += `  Đang thu thập dữ liệu (cần >= 10 phiên)...\n`;
  }

  // === DỰ ĐOÁN ===
  out += `${SEP2}\n  Phiếu bầu: TAI=${vt} XIU=${vx} Trung=${vn}  Đồng thuận: ${(agree*100).toFixed(0)}%\n${SEP}\n`;

  const margin = Math.abs(agg.taiP - agg.xiuP);
  const conf = Math.min(1.0, agree * 0.5 + margin + (agg.patternBoost || 0) * 0.1);
  const confLabel = conf >= 0.65 ? 'CAO' : (conf >= 0.45 ? 'TRUNG BÌNH' : 'THẤP');

  const isBroken = finalPred !== agg.pred;
  out += `  >> DỰ ĐOÁN PHIÊN TIẾP: ${finalPred} ${isBroken ? '(ĐÃ BẺ)' : ''} <<\n`;
  out += `  TAI [${bar(agg.taiP, 15)}] ${pct(agg.taiP)}\n`;
  out += `  XIU [${bar(agg.xiuP, 15)}] ${pct(agg.xiuP)}\n`;
  out += `  Raw: ${agg.rawScore.toFixed(4)}   Tin cậy: ${confLabel} (${pct(conf)})\n${SEP}\n`;

  // === THỐNG KÊ CHÍNH XÁC ===
  const acc = calcAccuracy();
  out += '  [ THỐNG KÊ DỰ ĐOÁN ]\n';
  if (acc.correct + acc.wrong > 0) {
    out += `  Tổng: ${acc.correct+acc.wrong}  Đúng: ${acc.correct}  Sai: ${acc.wrong}   Chính xác: ${pct(acc.acc)}\n`;
    out += `  [${bar(acc.acc, 20)}]\n`;
    // Streak chính xác gần đây
    const recent5 = state.resolvedPredictions.slice(-5);
    const streak5 = recent5.filter(x => x.correct).length;
    out += `  5 phiên gần nhất: ${streak5}/5 đúng\n`;
  } else {
    out += '  Chưa có dữ liệu (cần >= 1 phiên đã kiểm chứng)\n';
  }

  out += `${SEP2}\n  ${'PHIÊN'.padEnd(10)} ${'THỰC TẾ'.padEnd(8)} ${'DỰ ĐOÁN'.padEnd(10)} ĐÁNH GIÁ\n${SEP2}\n`;
  const recent = getRecentLog(15);
  if (recent.length) {
    for (const e of recent) {
      const ok = e.correct ? 'ĐÚNG [✓]' : 'SAI  [✗]';
      const manip = e.manipDetected ? '[⚠]' : '   ';
      out += `  ${String(e.phien).padEnd(10)} ${e.actual.padEnd(8)} ${(e.finalPredicted||e.predicted).padEnd(10)} ${ok} ${manip}\n`;
    }
  } else {
    out += '  (Đang chờ phiên đầu tiên hoàn tất...)\n';
  }

  out += `${SEP}\n  20 phiên: `;
  const hist20 = (data.history || []).slice(0, 20);
  out += hist20.map(s => s.result[0] + s.point).join(' ') + '\n';
  out += `  [Kiểm tra mỗi ${POLL_INTERVAL/1000}s | V4 Adaptive Learning]\n${SEP}`;
  return out;
}

// ============================================================
//  EXPRESS SERVER
// ============================================================
const app = express();
app.use(express.static('public'));

app.get('/', (req, res) => {
  const render = state.latestRender || 'Đang chờ dữ liệu lần đầu...';
  res.send(`<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MD5 Predictor V4</title>
  <style>
    body { background: #000; color: #0f0; font-family: 'Courier New', monospace;
           white-space: pre; margin: 10px; font-size: 13px; line-height: 1.4; }
    .warning { color: #ff4444; font-weight: bold; }
    .broken  { color: #ffaa00; }
  </style>
  <meta http-equiv="refresh" content="3">
</head>
<body>${render
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/\n/g,'<br>')
    .replace(/ /g,'&nbsp;')
    .replace(/!!!.*!!!/g, m => `<span class="warning">${m}</span>`)
    .replace(/\*\*.*\*\*/g, m => `<span class="broken">${m}</span>`)
}</body>
</html>`);
});

app.get('/api', (req, res) => {
  res.json({
    latestPrediction: state.latestPrediction,
    finalPrediction: state.finalPrediction,
    pendingPrediction: state.pendingPrediction,
    taiP: state.taiP,
    xiuP: state.xiuP,
    detail: state.detail,
    rawScore: state.rawScore,
    accuracy: state.accuracy,
    recentLog: state.recentLog,
    lastPhien: state.lastPhien,
    manipReport: state.manipReport,
    patternPred: state.patternPred,
    learningPhase: state.learningPhase,
    adaptiveLearning: state.adaptiveLearningStats,
    streakInfo: state.streakInfo
  });
});

// Reset weights (debug)
app.get('/reset-weights', (req, res) => {
  try { fs.unlinkSync(WEIGHT_SAVE_FILE); } catch(e) {}
  res.json({ ok: true, msg: 'Weights reset. Restart server.' });
});

app.listen(PORT, () => {
  console.log(`Server V4 đang chạy tại http://localhost:${PORT}`);
  mainLoop().catch(err => console.error('Lỗi vòng lặp:', err));
});