/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║        MD5 PREDICTOR V5 — ELITE ADAPTIVE ENGINE             ║
 * ║   Ensemble Learning + Smart Auto-Correction + Deep Analysis ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * Nâng cấp chính so với V4:
 *  1. ENSEMBLE VOTING  — 5 model con độc lập, kết hợp có trọng số
 *  2. ONLINE GRADIENT DESCENT — cập nhật trọng số sau MỖI phiên
 *  3. AUTO-CORRECTION ENGINE — phát hiện drift, tự bẻ khi cần
 *  4. BAYESIAN STREAK ANALYSIS — phân tích cầu có xác suất Bayes
 *  5. MARKOV CHAIN bậc 1, 2, 3 song song
 *  6. MOMENTUM SYSTEM — hạn chế bẻ hướng liên tục gây mất điểm
 *  7. CONFIDENCE GATING — chỉ bẻ khi điểm tin cậy đủ cao
 *  8. COLD START PROTECTION — 30 phiên đầu học trước khi kích hoạt bẻ
 *  9. PERFORMANCE CIRCUIT BREAKER — tắt tính năng khi liên tục sai
 * 10. FULL WEB DASHBOARD — giao diện đẹp, cập nhật real-time
 */

'use strict';

const express = require('express');
const crypto  = require('crypto');
const fetch   = require('node-fetch');
const fs      = require('fs');

// ══════════════════════════ CONFIG ══════════════════════════════
const API_URL          = "https://treo-lc79.onrender.com";
const FETCH_TIMEOUT    = 10000;
const POLL_INTERVAL    = 2000;
const HISTORY_WINDOW   = 120;
const MAX_TRACK        = 500;
const PORT             = process.env.PORT || 3000;
const STATE_FILE       = './v5_state.json';
const COLD_START_MIN   = 30;   // phiên tối thiểu trước khi bẻ được phép
const BREAK_THRESHOLD  = 0.60; // xác suất để kích hoạt auto-correction
const CIRCUIT_TRIP_SEQ = 5;    // sai liên tiếp bao nhiêu → trip circuit breaker

process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

// ══════════════════════════ UTILITIES ═══════════════════════════
const sleep = ms => new Promise(r => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const sigmoid = (x, c = 0.5, s = 8) => 1 / (1 + Math.exp(-s * (x - c)));
const pct    = v => (v * 100).toFixed(1) + '%';
const opp    = r => r === 'TAI' ? 'XIU' : 'TAI';

function md5Of(text)       { return crypto.createHash('md5').update(String(text)).digest('hex'); }
function md5NumVal(h)      { return parseInt(h.slice(0, 8), 16) / 0xFFFFFFFF; }
function md5BitSum(h)      { return (BigInt('0x' + h).toString(2).split('1').length - 1) / 128; }
function md5ByteMean(h)    { let s = 0; for (let i = 0; i < 32; i += 2) s += parseInt(h.substr(i,2),16); return s / (16*255); }
function md5Entropy(h)     { const f={}; for (const c of h) f[c]=(f[c]||0)+1; let e=0; for (const c in f){const p=f[c]/32; e-=p*Math.log2(p);} return e/4; }
function md5NibbleBias(h)  { return [...h].filter(c=>'89abcdef'.includes(c)).length/h.length; }
function md5Xor(h)         { return (parseInt(h[0],16)^parseInt(h[31],16))/15; }
function md5SumMod(h)      { let s=0; for(let i=0;i<32;i+=2)s+=parseInt(h.substr(i,2),16); return (s%256)/255; }
function md5EvenOdd(h)     { let ev=0; for(let i=0;i<32;i+=2){const b=parseInt(h.substr(i,2),16); if(b%2===0)ev++;} return ev/16; }

// ══════════════════════════ PERSISTENCE ═════════════════════════
function loadState() {
  try {
    if (fs.existsSync(STATE_FILE)) {
      const raw = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
      console.log(`[V5] Đã tải ${raw.totalResolved || 0} phiên lịch sử từ file.`);
      return raw;
    }
  } catch(e) { console.log('[V5] Bắt đầu mới (không có file state).'); }
  return null;
}

function saveState(data) {
  try { fs.writeFileSync(STATE_FILE, JSON.stringify(data, null, 2)); }
  catch(e) { console.error('[V5] Không lưu được state:', e.message); }
}

// ══════════════════════════════════════════════════════════════════
//  MODULE 1: ONLINE GRADIENT DESCENT WEIGHT MANAGER
//  Cập nhật trọng số sau MỖI phiên (không phải mỗi 5 phiên)
//  Dùng momentum để tránh oscillation
// ══════════════════════════════════════════════════════════════════
class OnlineWeightManager {
  constructor(featureKeys) {
    this.keys = featureKeys;
    this.lr   = 0.03;   // learning rate ban đầu
    this.momentum = 0.9;
    this.minW = 0.01;
    this.maxW = 0.40;
    this.weights    = {};
    this.velocity   = {};
    this.accuracy   = {};  // accuracy trượt mỗi feature
    this.updateN    = 0;

    // Khởi tạo đều
    const initW = 1 / featureKeys.length;
    for (const k of featureKeys) {
      this.weights[k]  = initW;
      this.velocity[k] = 0;
      this.accuracy[k] = 0.5; // giả sử 50% ban đầu
    }
  }

  restore(saved) {
    if (!saved) return;
    for (const k of this.keys) {
      if (saved.weights?.[k])  this.weights[k]  = saved.weights[k];
      if (saved.velocity?.[k]) this.velocity[k] = saved.velocity[k];
      if (saved.accuracy?.[k]) this.accuracy[k] = saved.accuracy[k];
    }
    this.updateN = saved.updateN || 0;
  }

  // Gradient descent với binary cross-entropy loss
  update(featureValues, predicted, actual) {
    const correctLabel = actual === 'TAI' ? 1 : 0;

    // Tính raw score từ weights hiện tại
    let score = 0, tw = 0;
    for (const k of this.keys) {
      if (featureValues[k] !== undefined) {
        score += this.weights[k] * featureValues[k];
        tw    += this.weights[k];
      }
    }
    score = tw > 0 ? score / tw : 0.5;
    const prob = sigmoid(score);

    // Gradient của BCE loss: dL/dprob = -(y/p - (1-y)/(1-p))
    const grad = prob - correctLabel; // simplified gradient

    // Adaptive learning rate (decay nhẹ theo số lần update)
    const lrEff = this.lr / (1 + this.updateN * 0.0002);

    for (const k of this.keys) {
      if (featureValues[k] === undefined) continue;

      // Gradient cho weight k: grad * feature_value
      const gk = grad * (featureValues[k] - 0.5); // centered

      // Momentum update
      this.velocity[k] = this.momentum * this.velocity[k] - lrEff * gk;
      this.weights[k]  = clamp(this.weights[k] + this.velocity[k], this.minW, this.maxW);

      // Cập nhật accuracy trượt (EMA)
      const featureCorrect = (featureValues[k] > 0.5) === (actual === 'TAI');
      this.accuracy[k] = 0.95 * this.accuracy[k] + 0.05 * (featureCorrect ? 1 : 0);
    }

    // Normalize weights để tổng = 1
    const sum = Object.values(this.weights).reduce((a,b)=>a+b, 0);
    for (const k of this.keys) this.weights[k] /= sum;

    this.updateN++;
  }

  toJSON() {
    return { weights: this.weights, velocity: this.velocity, accuracy: this.accuracy, updateN: this.updateN };
  }
}

// ══════════════════════════════════════════════════════════════════
//  MODULE 2: ENSEMBLE — 5 model con độc lập
//  Mỗi model có vùng chuyên sâu khác nhau
// ══════════════════════════════════════════════════════════════════
class EnsembleEngine {
  constructor() {
    // Trọng số của mỗi model trong ensemble, cập nhật online
    this.modelWeights = {
      MD5:     0.20,  // Model dựa trên hash MD5
      MARKOV:  0.20,  // Markov chains bậc 1,2,3
      PATTERN: 0.20,  // Pattern matching có nhớ
      STREAK:  0.20,  // Phân tích cầu Bayesian
      STAT:    0.20,  // Thống kê tần suất + chu kỳ
    };
    this.modelAccuracy = {
      MD5: 0.5, MARKOV: 0.5, PATTERN: 0.5, STREAK: 0.5, STAT: 0.5
    };
    this.modelHistory = { MD5: [], MARKOV: [], PATTERN: [], STREAK: [], STAT: [] };
  }

  restore(saved) {
    if (!saved) return;
    if (saved.modelWeights)   this.modelWeights   = { ...this.modelWeights,   ...saved.modelWeights };
    if (saved.modelAccuracy)  this.modelAccuracy  = { ...this.modelAccuracy,  ...saved.modelAccuracy };
    if (saved.modelHistory)   this.modelHistory   = { ...this.modelHistory,   ...saved.modelHistory };
  }

  // Cập nhật performance của từng model, điều chỉnh trọng số
  update(predictions, actual) {
    for (const [name, pred] of Object.entries(predictions)) {
      if (!pred) continue;
      const correct = pred === actual;
      this.modelHistory[name].push(correct);
      if (this.modelHistory[name].length > 50)
        this.modelHistory[name] = this.modelHistory[name].slice(-50);

      // Accuracy EMA
      this.modelAccuracy[name] = 0.92 * this.modelAccuracy[name] + 0.08 * (correct ? 1 : 0);
    }

    // Rebalance ensemble weights dựa trên accuracy (softmax)
    const exps = {};
    let sumExp = 0;
    for (const [name, acc] of Object.entries(this.modelAccuracy)) {
      exps[name] = Math.exp((acc - 0.5) * 8);
      sumExp += exps[name];
    }
    for (const name of Object.keys(this.modelWeights)) {
      this.modelWeights[name] = exps[name] / sumExp;
    }
  }

  // Kết hợp nhiều dự đoán có trọng số → điểm tổng hợp
  combine(modelProbs) {
    let score = 0, tw = 0;
    for (const [name, prob] of Object.entries(modelProbs)) {
      const w = this.modelWeights[name] || 0;
      score += w * prob;
      tw    += w;
    }
    return tw > 0 ? score / tw : 0.5;
  }

  toJSON() {
    return {
      modelWeights:  this.modelWeights,
      modelAccuracy: this.modelAccuracy,
      modelHistory:  this.modelHistory
    };
  }
}

// ══════════════════════════════════════════════════════════════════
//  MODULE 3: MARKOV CHAINS bậc 1, 2, 3
// ══════════════════════════════════════════════════════════════════
class MarkovEngine {
  constructor() {
    this.order1 = {};   // { 'TAI': { TAI: n, XIU: n } }
    this.order2 = {};   // { 'TAI_TAI': { TAI: n, XIU: n } }
    this.order3 = {};   // { 'TAI_TAI_XIU': { TAI: n, XIU: n } }
    this.totalTrained = 0;
  }

  restore(saved) {
    if (!saved) return;
    if (saved.order1) this.order1 = saved.order1;
    if (saved.order2) this.order2 = saved.order2;
    if (saved.order3) this.order3 = saved.order3;
    this.totalTrained = saved.totalTrained || 0;
  }

  train(history) {
    // Train từ toàn bộ history (không cần nhiều — chỉ cập nhật delta)
    this.order1 = {}; this.order2 = {}; this.order3 = {};
    const R = history.map(h => h.result);

    for (let i = 0; i < R.length - 1; i++) {
      const next = R[i]; // history[0] là mới nhất → next ở đây là "before" cho mẫu
      // Đảo ngược: history[i+1] → history[i]
      const prev1 = R[i+1];
      const key1 = prev1;
      if (!this.order1[key1]) this.order1[key1] = { TAI: 0, XIU: 0 };
      this.order1[key1][next]++;

      if (i + 2 < R.length) {
        const prev2 = R[i+2];
        const key2 = `${prev2}_${prev1}`;
        if (!this.order2[key2]) this.order2[key2] = { TAI: 0, XIU: 0 };
        this.order2[key2][next]++;
      }

      if (i + 3 < R.length) {
        const prev3 = R[i+3];
        const key3 = `${prev3}_${R[i+2]}_${prev1}`;
        if (!this.order3[key3]) this.order3[key3] = { TAI: 0, XIU: 0 };
        this.order3[key3][next]++;
      }
    }
    this.totalTrained = R.length;
  }

  // Trả về xác suất TAI cho phiên tiếp theo
  predict(recentResults) {
    if (recentResults.length < 1) return 0.5;

    const scores = [];

    // Order 1
    const k1 = recentResults[0];
    if (this.order1[k1]) {
      const { TAI=0, XIU=0 } = this.order1[k1];
      if (TAI + XIU >= 3) scores.push({ prob: TAI/(TAI+XIU), w: 1.0 });
    }

    // Order 2
    if (recentResults.length >= 2) {
      const k2 = `${recentResults[1]}_${recentResults[0]}`;
      if (this.order2[k2]) {
        const { TAI=0, XIU=0 } = this.order2[k2];
        if (TAI + XIU >= 3) scores.push({ prob: TAI/(TAI+XIU), w: 2.0 });
      }
    }

    // Order 3
    if (recentResults.length >= 3) {
      const k3 = `${recentResults[2]}_${recentResults[1]}_${recentResults[0]}`;
      if (this.order3[k3]) {
        const { TAI=0, XIU=0 } = this.order3[k3];
        if (TAI + XIU >= 2) scores.push({ prob: TAI/(TAI+XIU), w: 3.5 }); // ưu tiên order cao hơn
      }
    }

    if (!scores.length) return 0.5;
    const sw = scores.reduce((a,s)=>a+s.w, 0);
    return scores.reduce((a,s)=>a+s.prob*s.w, 0) / sw;
  }

  toJSON() {
    return { order1: this.order1, order2: this.order2, order3: this.order3, totalTrained: this.totalTrained };
  }
}

// ══════════════════════════════════════════════════════════════════
//  MODULE 4: ADVANCED PATTERN MEMORY
//  Bộ nhớ pattern dài hơn (3-7), với decay theo thời gian
// ══════════════════════════════════════════════════════════════════
class AdvancedPatternMemory {
  constructor() {
    this.patterns = {};   // { key: { TAI: w, XIU: w } } — weighted counts
    this.LENGTHS  = [3, 4, 5, 6, 7];
    this.DECAY    = 0.98; // mỗi lần train, decay các count cũ
  }

  restore(saved) {
    if (saved?.patterns) this.patterns = saved.patterns;
  }

  encode(results, len) {
    return results.slice(0, len).map(r => r[0]).join('');
  }

  train(history) {
    // Decay tất cả existing counts
    for (const p of Object.values(this.patterns)) {
      p.TAI *= this.DECAY;
      p.XIU *= this.DECAY;
    }

    const R = history.map(h => h.result);
    for (const len of this.LENGTHS) {
      for (let i = 0; i < R.length - len - 1; i++) {
        // Pattern là các kết quả TRƯỚC kết quả cần dự đoán
        const patternResults = R.slice(i + 1, i + 1 + len);
        const next = R[i];
        const key = this.encode(patternResults, len);
        if (!this.patterns[key]) this.patterns[key] = { TAI: 0, XIU: 0 };
        // Trọng số gần đây cao hơn
        const recencyBoost = Math.exp(-i * 0.05);
        this.patterns[key][next] += recencyBoost;
      }
    }
  }

  predict(recentResults) {
    let bestScore = 0;
    let bestProb  = 0.5;
    let bestKey   = null;
    let bestStats = null;

    for (const len of [...this.LENGTHS].reverse()) { // ưu tiên dài hơn
      if (recentResults.length < len) continue;
      const key = this.encode(recentResults, len);
      const p   = this.patterns[key];
      if (!p) continue;

      const total = p.TAI + p.XIU;
      if (total < 2) continue; // cần ít nhất 2 weighted sample

      const taiRate   = p.TAI / total;
      const confidence = Math.abs(taiRate - 0.5) * 2; // 0..1
      const score     = confidence * Math.log(1 + total) * len; // dài + nhiều mẫu = tốt hơn

      if (score > bestScore) {
        bestScore = score;
        bestProb  = taiRate;
        bestKey   = key;
        bestStats = { TAI: Math.round(p.TAI), XIU: Math.round(p.XIU), total: Math.round(total) };
      }
    }

    return {
      prob:       bestProb,
      prediction: bestProb > 0.5 ? 'TAI' : (bestProb < 0.5 ? 'XIU' : null),
      confidence: Math.abs(bestProb - 0.5) * 2,
      pattern:    bestKey,
      stats:      bestStats
    };
  }

  getTotalPatterns() { return Object.keys(this.patterns).length; }

  toJSON() { return { patterns: this.patterns }; }
}

// ══════════════════════════════════════════════════════════════════
//  MODULE 5: BAYESIAN STREAK ANALYZER
//  Phân tích cầu dùng Bayes prior để tránh overfit cầu ngắn
// ══════════════════════════════════════════════════════════════════
class BayesianStreakAnalyzer {
  constructor() {
    // Prior: xác suất tiếp tục cầu theo độ dài (học từ dữ liệu)
    this.continuePrior = {
      1: { count: 0, continue: 0 },
      2: { count: 0, continue: 0 },
      3: { count: 0, continue: 0 },
      4: { count: 0, continue: 0 },
      5: { count: 0, continue: 0 },
      6: { count: 0, continue: 0 },
    };
    this.PRIOR_ALPHA = 3;  // pseudo-count prior (assume 50% by default)
    this.PRIOR_BETA  = 3;
  }

  restore(saved) {
    if (saved?.continuePrior) this.continuePrior = saved.continuePrior;
  }

  train(history) {
    // Học tỷ lệ tiếp cầu vs đảo cầu theo từng độ dài
    const R = history.map(h => h.result);
    for (let i = 0; i < R.length - 1; i++) {
      let len = 1;
      while (i + len < R.length && R[i + len] === R[i]) len++;
      len = Math.min(len, 6);
      if (!this.continuePrior[len]) this.continuePrior[len] = { count: 0, continue: 0 };
      // Xem kết quả trước R[i] (nếu có) là tiếp tục hay đảo
      if (i > 0) {
        this.continuePrior[len].count++;
        if (R[i-1] === R[i]) this.continuePrior[len].continue++;
      }
      i += len - 1; // nhảy qua cầu
    }
  }

  // Trả về xác suất TAI cho phiên tiếp theo
  predict(streakCur, streakLen) {
    if (!streakCur) return 0.5;

    const lenKey = Math.min(streakLen, 6);
    const prior  = this.continuePrior[lenKey] || { count: 0, continue: 0 };

    // Bayesian estimate: (alpha + success) / (alpha + beta + total)
    const alpha   = this.PRIOR_ALPHA;
    const beta    = this.PRIOR_BETA;
    const pCont   = (alpha + prior.continue) / (alpha + beta + prior.count);

    // Nếu tiếp tục → cùng loại, nếu đảo → loại kia
    if (streakCur === 'TAI') {
      return pCont; // prob(TAI tiếp theo)
    } else {
      return 1 - pCont; // prob(TAI tiếp theo khi XIU đang cầu)
    }
  }

  toJSON() { return { continuePrior: this.continuePrior }; }
}

// ══════════════════════════════════════════════════════════════════
//  MODULE 6: MANIPULATION DETECTOR (nâng cấp)
//  Phát hiện can thiệp có độ nhạy cao hơn, ít false positive hơn
// ══════════════════════════════════════════════════════════════════
class ManipulationDetectorV2 {
  constructor() {
    this.WINDOW     = 30;
    this.score      = 0;
    this.signals    = [];
    this.detected   = false;
    this.type       = null;
    this.confidence = 0;
  }

  analyze(history) {
    if (history.length < 15) return this._report(false, [], 0);

    const R = history.map(h => h.result).slice(0, this.WINDOW);
    const P = history.map(h => h.point).slice(0, this.WINDOW);
    const signals = [];

    // ── Signal 1: Tần suất lệch mạnh ──────────────────────────
    const taiCount = R.filter(r => r === 'TAI').length;
    const ratio    = taiCount / R.length;
    if (ratio > 0.72 || ratio < 0.28) {
      signals.push({
        type: 'FREQ_BIAS', severity: 'HIGH',
        desc: `Tần suất TAI lệch: ${pct(ratio)}`,
        score: clamp(Math.abs(ratio - 0.5) * 3, 0, 1)
      });
    }

    // ── Signal 2: Cầu siêu dài (≥ 8) ─────────────────────────
    let maxStreak = 1, cur = 1;
    for (let i = 1; i < R.length; i++) {
      if (R[i] === R[i-1]) { cur++; maxStreak = Math.max(maxStreak, cur); }
      else cur = 1;
    }
    if (maxStreak >= 8) {
      signals.push({
        type: 'LONG_STREAK', severity: 'MEDIUM',
        desc: `Cầu siêu dài: ${maxStreak} phiên liên tiếp`,
        score: clamp((maxStreak - 6) * 0.12, 0, 0.7)
      });
    }

    // ── Signal 3: Pattern A^n B^m A^n có chủ đích ────────────
    const flip = this._detectFlip(R);
    if (flip.detected) {
      signals.push({
        type: 'FLIP_PATTERN', severity: 'HIGH',
        desc: `Đảo cầu có chủ đích: ${flip.desc}`,
        score: 0.65
      });
    }

    // ── Signal 4: Điểm xúc xắc lệch ─────────────────────────
    if (P.length >= 15) {
      const avg = P.reduce((a,b)=>a+b,0) / P.length;
      const dev = Math.abs(avg - 10.5);
      if (dev > 2.2) {
        signals.push({
          type: 'DICE_SKEW', severity: 'LOW',
          desc: `Điểm TB xúc xắc lệch: ${avg.toFixed(1)} (chuẩn 10.5)`,
          score: clamp(dev / 5.5, 0, 0.6)
        });
      }
    }

    // ── Signal 5: Alternating pattern cứng (TXTTXTTX...) ─────
    let altCount = 0;
    for (let i = 0; i < Math.min(R.length-2, 20); i++) {
      if (R[i] !== R[i+1] && R[i] === R[i+2]) altCount++;
    }
    if (altCount / Math.min(R.length-2, 20) > 0.70) {
      signals.push({
        type: 'STRICT_ALT', severity: 'MEDIUM',
        desc: `Cầu đảo cứng bất thường: ${pct(altCount/Math.min(R.length-2,20))}`,
        score: 0.55
      });
    }

    // Tổng hợp score với diminishing returns
    let totalScore = 0;
    for (let i = 0; i < signals.length; i++) {
      totalScore += signals[i].score / (1 + i * 0.3); // giảm dần
    }
    totalScore = clamp(totalScore, 0, 1);

    this.score    = totalScore;
    this.detected = totalScore > 0.45;
    this.type     = signals[0]?.type || null;
    this.signals  = signals;
    this.confidence = totalScore;

    return this._report(this.detected, signals, totalScore);
  }

  _detectFlip(R) {
    let i = 0;
    while (i < R.length - 7) {
      const base = R[i];
      let runA = 0;
      while (i + runA < R.length && R[i+runA] === base) runA++;
      if (runA >= 3) {
        const op = opp(base);
        let runB = 0;
        while (i + runA + runB < R.length && R[i+runA+runB] === op) runB++;
        if (runB >= 3) return { detected: true, desc: `${base}×${runA} → ${op}×${runB}` };
      }
      i += Math.max(runA, 1);
    }
    return { detected: false };
  }

  _report(detected, signals, score) {
    return { isManipulated: detected, signals, score, type: this.type };
  }

  // Gợi ý chiến lược bẻ thông minh hơn
  getCounterStrategy(currentPred, manipReport, momentum) {
    if (!manipReport.isManipulated) return { pred: currentPred, broke: false, reason: null };

    const score   = manipReport.score;
    const types   = manipReport.signals.map(s => s.type);
    const penalty = momentum.breakCount > 2 ? 0.10 : 0; // nếu bẻ nhiều lần liên tiếp → giảm threshold

    // FLIP_PATTERN: bẻ nếu score > 0.55
    if (types.includes('FLIP_PATTERN') && score > 0.55 + penalty) {
      return { pred: opp(currentPred), broke: true, reason: 'FLIP_PATTERN' };
    }

    // STRICT_ALT: dự đoán theo pattern đảo
    if (types.includes('STRICT_ALT') && score > 0.50 + penalty) {
      return { pred: opp(currentPred), broke: true, reason: 'STRICT_ALT' };
    }

    // FREQ_BIAS: chỉ bẻ khi rất rõ ràng
    if (types.includes('FREQ_BIAS') && score > 0.65 + penalty) {
      return { pred: opp(currentPred), broke: true, reason: 'FREQ_BIAS' };
    }

    return { pred: currentPred, broke: false, reason: null };
  }
}

// ══════════════════════════════════════════════════════════════════
//  MODULE 7: AUTO-CORRECTION ENGINE (bẻ tự động thông minh)
//  Phát hiện drift trong accuracy gần đây → tự điều chỉnh
// ══════════════════════════════════════════════════════════════════
class AutoCorrectionEngine {
  constructor() {
    this.WINDOW_SHORT = 8;   // cửa sổ ngắn để phát hiện nhanh
    this.WINDOW_LONG  = 20;  // cửa sổ dài để xác nhận xu hướng
    this.BREAK_CONF   = 0.62; // ngưỡng để bẻ (cần xác suất cao)
    this.MAX_SEQ_BREAK = 3;  // tối đa bẻ liên tiếp bao nhiêu lần
    this.breakCount   = 0;   // số lần bẻ liên tiếp gần đây
    this.lastBreak    = false;
    this.circuitOpen  = false;  // circuit breaker (tắt bẻ khi liên tục sai)
    this.circuitCooldown = 0;
  }

  restore(saved) {
    if (!saved) return;
    this.breakCount     = saved.breakCount     || 0;
    this.lastBreak      = saved.lastBreak      || false;
    this.circuitOpen    = saved.circuitOpen    || false;
    this.circuitCooldown= saved.circuitCooldown|| 0;
  }

  // Phân tích accuracy gần đây, trả về có nên bẻ không
  shouldBreak(resolved, rawPred, rawProb) {
    const total = resolved.length;

    // Chưa đủ dữ liệu → không bẻ
    if (total < COLD_START_MIN) {
      return { break: false, reason: 'COLD_START', corrected: rawPred };
    }

    // Circuit breaker đang mở → không bẻ
    if (this.circuitOpen) {
      this.circuitCooldown--;
      if (this.circuitCooldown <= 0) {
        this.circuitOpen = false;
        console.log('[AutoCorrect] Circuit breaker đóng lại.');
      }
      return { break: false, reason: 'CIRCUIT_OPEN', corrected: rawPred };
    }

    // Đang bẻ liên tiếp quá nhiều → dừng
    if (this.breakCount >= this.MAX_SEQ_BREAK) {
      return { break: false, reason: 'MAX_BREAK_REACHED', corrected: rawPred };
    }

    // Tính accuracy ngắn và dài
    const shortWindow = resolved.slice(-this.WINDOW_SHORT);
    const longWindow  = resolved.slice(-this.WINDOW_LONG);

    if (shortWindow.length < 5) return { break: false, reason: 'INSUFFICIENT', corrected: rawPred };

    const shortAcc = shortWindow.filter(x => x.correct).length / shortWindow.length;
    const longAcc  = longWindow.filter(x => x.correct).length / longWindow.length;

    // Tính chuỗi sai liên tiếp gần nhất
    let seqWrong = 0;
    for (let i = resolved.length - 1; i >= 0; i--) {
      if (!resolved[i].correct) seqWrong++;
      else break;
    }

    // Trip circuit breaker nếu sai quá nhiều liên tiếp
    if (seqWrong >= CIRCUIT_TRIP_SEQ) {
      this.circuitOpen     = true;
      this.circuitCooldown = 8;
      this.breakCount      = 0;
      console.log(`[AutoCorrect] Circuit breaker MỞ — ${seqWrong} sai liên tiếp!`);
      return { break: false, reason: 'CIRCUIT_TRIPPED', corrected: rawPred };
    }

    // ── Điều kiện bẻ ────────────────────────────────────────
    // Cần: accuracy ngắn thấp VÀ mô hình có confidence đủ cao ở chiều ngược
    const reversedProb = 1 - rawProb;
    const shouldBreak  = (
      shortAcc < 0.35 &&            // accuracy ngắn dưới 35%
      longAcc  < 0.48 &&            // xu hướng dài cũng không tốt
      reversedProb >= this.BREAK_CONF // phía ngược có confidence ≥ 62%
    );

    if (shouldBreak) {
      return {
        break: true,
        reason: `ACC_DRIFT:short=${pct(shortAcc)},long=${pct(longAcc)}`,
        corrected: opp(rawPred),
        seqWrong
      };
    }

    // Không bẻ
    return { break: false, reason: 'NOMINAL', corrected: rawPred };
  }

  // Gọi sau mỗi phiên để cập nhật state
  updateAfterResult(breakUsed, wasCorrect) {
    if (breakUsed) {
      if (wasCorrect) {
        this.breakCount = Math.max(0, this.breakCount - 1); // bẻ thành công → giảm count
      } else {
        this.breakCount++; // bẻ sai → tăng count
      }
      this.lastBreak = true;
    } else {
      this.breakCount = Math.max(0, this.breakCount - 0.5); // tự phục hồi
      this.lastBreak  = false;
    }
  }

  toJSON() {
    return {
      breakCount:      this.breakCount,
      lastBreak:       this.lastBreak,
      circuitOpen:     this.circuitOpen,
      circuitCooldown: this.circuitCooldown
    };
  }
}

// ══════════════════════════════════════════════════════════════════
//  STAT ENGINE (nâng cấp — giữ nguyên core, thêm features)
// ══════════════════════════════════════════════════════════════════
class StatEngineV2 {
  constructor(history) {
    this.R = history.map(s => s.result);
    this.P = history.map(s => s.point);
    this.D = history.map(s => s.dices);
    this.N = history.length;
  }

  // Tần suất có trọng số gần đây cao hơn
  frequencyWeighted(w = 30) {
    const tail = this.R.slice(0, w);
    if (!tail.length) return 0.5;
    let taiScore = 0, total = 0;
    tail.forEach((r, i) => {
      const weight = Math.exp(-i * 0.06); // gần đây quan trọng hơn
      taiScore += weight * (r === 'TAI' ? 1 : 0);
      total    += weight;
    });
    return taiScore / total;
  }

  streakCurrent() {
    if (!this.R.length) return { cur: null, len: 0 };
    const cur = this.R[0];
    let len = 0;
    for (const r of this.R) { if (r === cur) len++; else break; }
    return { cur, len };
  }

  pointTrendNorm(w = 15) {
    const pts = this.P.slice(0, w);
    if (!pts.length) return 0.5;
    const avg = pts.reduce((a,b)=>a+b,0) / pts.length;
    return clamp((avg - 3) / 15, 0, 1);
  }

  diceFaceBias(w = 20) {
    const fc = {};
    this.D.slice(0, w).forEach(dice => dice.forEach(f => fc[f]=(fc[f]||0)+1));
    const total = Object.values(fc).reduce((a,b)=>a+b, 0);
    if (!total) return 0.5;
    return ((fc[4]||0) + (fc[5]||0) + (fc[6]||0)) / total;
  }

  // Chu kỳ tìm từ FFT-lite (autocorrelation đơn giản)
  periodicity(w = 60) {
    const binary = this.R.slice(0, w).map(r => r === 'TAI' ? 1 : -1);
    if (binary.length < 10) return 0.5;

    let bestLag = 2, bestCorr = 0;
    for (let lag = 2; lag <= Math.min(10, Math.floor(binary.length / 3)); lag++) {
      let corr = 0;
      for (let i = 0; i < binary.length - lag; i++) corr += binary[i] * binary[i + lag];
      corr /= (binary.length - lag);
      if (Math.abs(corr) > Math.abs(bestCorr)) { bestCorr = corr; bestLag = lag; }
    }

    // Dự đoán theo chu kỳ tốt nhất
    const nextInCycle = binary[bestLag % binary.length];
    return nextInCycle > 0 ? clamp(0.5 + Math.abs(bestCorr) * 0.4, 0, 1)
                           : clamp(0.5 - Math.abs(bestCorr) * 0.4, 0, 1);
  }

  // Biến động điểm gần đây (volatility)
  pointVolatility(w = 10) {
    const pts = this.P.slice(0, w);
    if (pts.length < 3) return 0.5;
    const avg = pts.reduce((a,b)=>a+b,0)/pts.length;
    const variance = pts.reduce((a,b)=>a+(b-avg)**2, 0)/pts.length;
    const std = Math.sqrt(variance);
    // Volatility cao → điểm dao động → xu hướng TAI/XIU ít ổn định
    return clamp(std / 4, 0, 1);
  }
}

// ══════════════════════════════════════════════════════════════════
//  MD5 ANALYSIS (thêm 2 layers mới)
// ══════════════════════════════════════════════════════════════════
function analyzeMd5V5(session) {
  const id    = session._id || '';
  const phien = String(session.phien || '');
  const mi    = md5Of(id);
  const mp    = md5Of(phien);
  const mc    = md5Of(id + phien);
  const md    = md5Of(phien + id);  // đảo ngược
  return {
    L1: md5NumVal(mi),
    L2: md5BitSum(mp),
    L3: md5ByteMean(mc),
    L4: md5Entropy(mi),
    L5: md5NibbleBias(mc),
    L6: md5Xor(mi),
    L7: md5SumMod(md),   // MỚI
    L8: md5EvenOdd(mc),  // MỚI
    _mi: mi, _mp: mp, _mc: mc
  };
}

// ══════════════════════════════════════════════════════════════════
//  FEATURE KEYS (tất cả features)
// ══════════════════════════════════════════════════════════════════
const ALL_FEATURE_KEYS = [
  'L1','L2','L3','L4','L5','L6','L7','L8',   // MD5 layers
  'A','B','C','D','E','F','G'                 // Stat features
];

// ══════════════════════════════════════════════════════════════════
//  GLOBAL INSTANCES
// ══════════════════════════════════════════════════════════════════
const weightManager = new OnlineWeightManager(ALL_FEATURE_KEYS);
const ensembleEng   = new EnsembleEngine();
const markovEng     = new MarkovEngine();
const patternMem    = new AdvancedPatternMemory();
const streakAnalyzer= new BayesianStreakAnalyzer();
const manipDetector = new ManipulationDetectorV2();
const autoCorrector = new AutoCorrectionEngine();

// ══════════════════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════════════════
const state = {
  pendingPrediction:  null,
  pendingRawProb:     0.5,
  pendingDetail:      null,
  pendingModelPreds:  null,
  pendingBreakUsed:   false,
  resolvedPredictions: [],
  lastPhien:          null,
  latestRender:       null,
  totalResolved:      0,

  // Untuk API
  latestPrediction: null,
  finalBreakReason: null,
  taiP: 0.5, xiuP: 0.5,
  manipReport:    null,
  patternResult:  null,
  streakInfo:     { cur: null, len: 0 },
  accuracy:       { correct: 0, wrong: 0, acc: 0 },
  recentLog:      [],
  learningPhase:  'COLD_START',
  autoCorrectInfo: null,
  ensembleWeights: {},
  featureWeights:  {},
  featureAccuracy: {},
};

// ══════════════════════════════════════════════════════════════════
//  PERSISTENCE — Load/Save
// ══════════════════════════════════════════════════════════════════
function restoreFromDisk() {
  const saved = loadState();
  if (!saved) return;

  weightManager.restore(saved.weightManager);
  ensembleEng.restore(saved.ensemble);
  markovEng.restore(saved.markov);
  patternMem.restore(saved.pattern);
  streakAnalyzer.restore(saved.streak);
  autoCorrector.restore(saved.autoCorrect);

  if (saved.resolvedPredictions) {
    state.resolvedPredictions = saved.resolvedPredictions.slice(-MAX_TRACK);
    state.totalResolved       = saved.totalResolved || state.resolvedPredictions.length;
  }
}

function saveToDisk() {
  saveState({
    weightManager:        weightManager.toJSON(),
    ensemble:             ensembleEng.toJSON(),
    markov:               markovEng.toJSON(),
    pattern:              patternMem.toJSON(),
    streak:               streakAnalyzer.toJSON(),
    autoCorrect:          autoCorrector.toJSON(),
    resolvedPredictions:  state.resolvedPredictions.slice(-MAX_TRACK),
    totalResolved:        state.totalResolved,
    savedAt:              new Date().toISOString()
  });
}

// ══════════════════════════════════════════════════════════════════
//  MAIN PREDICTION PIPELINE
// ══════════════════════════════════════════════════════════════════
function runPrediction(history) {
  const latest     = history[0];
  const md5        = analyzeMd5V5(latest);
  const statEng    = new StatEngineV2(history);
  const { cur: streakCur, len: streakLen } = statEng.streakCurrent();
  const recent10   = history.slice(0, 10).map(h => h.result);

  // ── Feature values (all normalized 0..1) ────────────────────
  const freqW = statEng.frequencyWeighted();
  const flip  = statEng.streakCurrent().len;
  // B: xác suất đảo cầu (streakLen dài → dễ đảo)
  const streakFlipProb = clamp(0.35 + flip * 0.06, 0, 0.90);
  const bVal = streakCur === 'TAI' ? (1 - streakFlipProb) : streakFlipProb;

  const features = {
    L1: md5.L1, L2: md5.L2, L3: md5.L3, L4: md5.L4,
    L5: md5.L5, L6: md5.L6, L7: md5.L7, L8: md5.L8,
    A: freqW,
    B: bVal,
    C: statEng.pointTrendNorm(),
    D: statEng.diceFaceBias(),
    E: statEng.periodicity(),
    F: 0.5, // placeholder — sẽ override bởi Markov
    G: statEng.pointVolatility(),
  };

  // ── Model 1: MD5 ─────────────────────────────────────────────
  const md5Keys = ['L1','L2','L3','L4','L5','L6','L7','L8'];
  const w = weightManager.weights;
  let md5Score = 0, md5W = 0;
  for (const k of md5Keys) {
    md5Score += (w[k] || 0.125) * features[k];
    md5W     += (w[k] || 0.125);
  }
  const md5Prob = sigmoid(md5W > 0 ? md5Score / md5W : 0.5);

  // ── Model 2: Markov ──────────────────────────────────────────
  const markovProb = markovEng.predict(recent10);
  features.F = markovProb;

  // ── Model 3: Pattern ─────────────────────────────────────────
  const patResult  = patternMem.predict(recent10);
  const patternProb= patResult.prob;

  // ── Model 4: Streak (Bayesian) ───────────────────────────────
  const streakProb = streakAnalyzer.predict(streakCur, streakLen);

  // ── Model 5: Stats ───────────────────────────────────────────
  const statKeys = ['A','B','C','D','E','G'];
  let statScore = 0, statW = 0;
  for (const k of statKeys) {
    statScore += (w[k] || 0.14) * features[k];
    statW     += (w[k] || 0.14);
  }
  const statProb = sigmoid(statW > 0 ? statScore / statW : 0.5);

  // ── Ensemble combine ─────────────────────────────────────────
  const modelProbs = {
    MD5:     md5Prob,
    MARKOV:  markovProb,
    PATTERN: patternProb,
    STREAK:  streakProb,
    STAT:    statProb,
  };
  const ensembleProb = ensembleEng.combine(modelProbs);

  // ── Feature-weighted score (full) ────────────────────────────
  let fullScore = 0, fullW = 0;
  for (const k of ALL_FEATURE_KEYS) {
    fullScore += (w[k] || 1/ALL_FEATURE_KEYS.length) * features[k];
    fullW     += (w[k] || 1/ALL_FEATURE_KEYS.length);
  }
  const fullProb = sigmoid(fullW > 0 ? fullScore / fullW : 0.5);

  // ── Final blend: ensemble 60% + full-feature 40% ─────────────
  const blendedProb = ensembleProb * 0.60 + fullProb * 0.40;

  const taiP = blendedProb;
  const xiuP = 1 - blendedProb;
  const rawPred = taiP >= xiuP ? 'TAI' : 'XIU';

  return {
    rawPred, taiP, xiuP,
    blendedProb, ensembleProb, fullProb,
    features, modelProbs,
    patResult,
    streakCur, streakLen,
    md5: { _mi: md5._mi, _mp: md5._mp, _mc: md5._mc }
  };
}

// ══════════════════════════════════════════════════════════════════
//  ACCURACY
// ══════════════════════════════════════════════════════════════════
function calcAccuracy(window = null) {
  const resolved = window ? state.resolvedPredictions.slice(-window) : state.resolvedPredictions;
  if (!resolved.length) return { correct: 0, wrong: 0, acc: 0, total: 0 };
  const correct = resolved.filter(x => x.correct).length;
  return { correct, wrong: resolved.length - correct, acc: correct / resolved.length, total: resolved.length };
}

// ══════════════════════════════════════════════════════════════════
//  RENDER — Terminal + Web dashboard
// ══════════════════════════════════════════════════════════════════
function bar(v, w = 15, filled = '#', empty = '-') {
  const n = clamp(Math.round(v * w), 0, w);
  return filled.repeat(n) + empty.repeat(w - n);
}
function arrow(v) {
  if (v > 0.56) return 'TAI↑';
  if (v < 0.44) return 'XIU↓';
  return ' ── ';
}

function buildTerminalRender(data, pred) {
  const SEP  = '═'.repeat(60);
  const SEP2 = '─'.repeat(60);
  const now  = new Date().toLocaleTimeString('vi-VN', { hour12: false });

  const latest  = data.history[0];
  const phien   = latest?.phien || '?';
  const dices   = latest?.dices || [];
  const point   = latest?.point || '?';
  const result  = latest?.result || '?';

  const acc     = calcAccuracy();
  const acc10   = calcAccuracy(10);
  const acc30   = calcAccuracy(30);
  const resolved= state.resolvedPredictions;
  let seqWrong  = 0, seqRight = 0;
  for (let i = resolved.length - 1; i >= 0; i--) {
    if (!resolved[i].correct) { seqRight=0; seqWrong++; } else break;
  }
  for (let i = resolved.length - 1; i >= 0; i--) {
    if (resolved[i].correct) { seqWrong=0; seqRight++; } else break;
  }

  const phaseMap = {
    'COLD_START': '[ KHỞI ĐỘNG — ĐANG HỌC ]',
    'LEARNING':   '[ ĐANG HỌC — TRỌNG SỐ ĐIỀU CHỈNH ]',
    'OPTIMIZED':  '[ TỐI ƯU — ĐẦY ĐỦ KHẢ NĂNG ]'
  };

  let out = '';
  out += `${SEP}\n  MD5 PREDICTOR V5 ELITE — ADAPTIVE ENSEMBLE    ${now}\n${SEP}\n`;
  out += `  ${phaseMap[state.learningPhase] || ''}\n${SEP2}\n`;
  out += `  Phiên   : ${phien}\n`;
  out += `  Xúc xắc : [${dices.join(' ')}]   Tổng: ${point}\n`;
  out += `  Kết quả : ${result}   |   Cầu: ${pred.streakCur || '?'} × ${pred.streakLen}\n`;
  out += `${SEP2}\n`;

  // Manipulation
  const mr = state.manipReport;
  if (mr?.isManipulated) {
    out += `  ⚠  CẢNH BÁO: PHÁT HIỆN NHÀ CÁI CAN THIỆP  ⚠\n`;
    out += `  Điểm nghi ngờ: ${pct(mr.score)}   Loại: ${mr.type}\n`;
    for (const s of mr.signals) out += `  >> [${s.severity}] ${s.desc}\n`;
    out += `${SEP2}\n`;
  }

  // MD5 Layers
  out += '  [ MD5 LAYERS ]\n';
  const layerDefs = [
    ['L1','NormVal  '],['L2','BitSum   '],['L3','ByteMean '],
    ['L4','Entropy  '],['L5','Nibble   '],['L6','XorChk   '],
    ['L7','SumMod   '],['L8','EvenOdd  ']
  ];
  const fw = weightManager.weights;
  for (const [k, lbl] of layerDefs) {
    const v = pred.features[k] ?? 0.5;
    out += `  ${k}:${lbl}[${bar(v,12)}] ${pct(v).padStart(6)}  ${arrow(v)}  w=${pct(fw[k]||0)}\n`;
  }

  // Stat Features
  out += `${SEP2}\n  [ STAT FEATURES ]\n`;
  const statDefs = [
    ['A','Freq(w)  '],['B','StreakFlp'],['C','PointTrn '],
    ['D','DiceBias '],['E','Periodic '],['F','Markov   '],['G','Volatil  ']
  ];
  for (const [k, lbl] of statDefs) {
    const v   = pred.features[k] ?? 0.5;
    const acc_ = weightManager.accuracy[k];
    const accS = acc_ !== undefined ? ` acc=${pct(acc_)}` : '';
    out += `  ${k}:${lbl}[${bar(v,12)}] ${pct(v).padStart(6)}  ${arrow(v)}  w=${pct(fw[k]||0)}${accS}\n`;
  }

  // Ensemble breakdown
  out += `${SEP2}\n  [ ENSEMBLE MODELS ]\n`;
  const modelDefs = ['MD5','MARKOV','PATTERN','STREAK','STAT'];
  for (const name of modelDefs) {
    const prob = pred.modelProbs[name] ?? 0.5;
    const ew   = ensembleEng.modelWeights[name] || 0;
    const ea   = ensembleEng.modelAccuracy[name] || 0.5;
    out += `  ${name.padEnd(8)}: [${bar(prob,10)}] ${pct(prob)}  w=${pct(ew)}  acc=${pct(ea)}\n`;
  }

  // Pattern
  out += `${SEP2}\n  [ PATTERN MEMORY — ${patternMem.getTotalPatterns()} mẫu ]\n`;
  const pr = state.patternResult;
  if (pr?.pattern) {
    out += `  Mẫu khớp: "${pr.pattern}"  Dự đoán: ${pr.prediction}  (tin cậy: ${pct(pr.confidence)})\n`;
    if (pr.stats) out += `  Lịch sử: TAI~${pr.stats.TAI} / XIU~${pr.stats.XIU} (n~${pr.stats.total})\n`;
  } else {
    out += `  Chưa đủ dữ liệu pattern.\n`;
  }

  // Auto-correct info
  const ai = state.autoCorrectInfo;
  if (ai) {
    out += `${SEP2}\n  [ AUTO-CORRECTION ]\n`;
    out += `  Trạng thái: ${autoCorrector.circuitOpen ? 'CIRCUIT OPEN ⛔' : 'HOẠT ĐỘNG ✓'}\n`;
    out += `  Break liên tiếp: ${Math.floor(autoCorrector.breakCount)}/${autoCorrector.MAX_SEQ_BREAK}\n`;
    if (ai.broke) out += `  ** BẺ: ${opp(state.latestPrediction)} → ${state.latestPrediction}  [${ai.reason}] **\n`;
  }

  // Prediction
  out += `${SEP2}\n`;
  const { taiP, xiuP } = state;
  const finalPred = state.latestPrediction || '?';
  const isBroken  = ai?.broke || mr?.isManipulated;
  const margin    = Math.abs(taiP - 0.5);
  const confLabel = margin > 0.15 ? 'CAO' : margin > 0.08 ? 'TRUNG BÌNH' : 'THẤP';

  out += `  >> DỰ ĐOÁN PHIÊN TIẾP: ${finalPred} ${isBroken ? '(ĐÃ BẺ ⚡)' : ''} <<\n`;
  out += `  TAI [${bar(taiP, 18)}] ${pct(taiP)}\n`;
  out += `  XIU [${bar(xiuP, 18)}] ${pct(xiuP)}\n`;
  out += `  Blend: ${pred.blendedProb.toFixed(4)}   Tin cậy: ${confLabel} (${pct(margin*2)})\n`;
  out += `${SEP}\n`;

  // Accuracy stats
  out += '  [ THỐNG KÊ CHÍNH XÁC ]\n';
  if (acc.total > 0) {
    out += `  Tổng: ${acc.total}  Đúng: ${acc.correct}  Sai: ${acc.wrong}   All: ${pct(acc.acc)}\n`;
    out += `  Gần10: ${pct(acc10.acc)}  Gần30: ${pct(acc30.acc)}\n`;
    out += `  [${bar(acc.acc, 24)}]\n`;
    const streak5 = resolved.slice(-5).filter(x=>x.correct).length;
    out += `  5 phiên gần nhất: ${streak5}/5  |  `;
    out += seqRight > 0 ? `Chuỗi đúng: ${seqRight}\n` : `Chuỗi sai: ${seqWrong}\n`;
  } else {
    out += '  Chưa có dữ liệu (cần ít nhất 1 phiên xác nhận)\n';
  }

  // Recent log
  out += `${SEP2}\n  ${'PHIÊN'.padEnd(12)} ${'THỰC TẾ'.padEnd(8)} ${'DỰ ĐOÁN'.padEnd(10)} KẾT QUẢ\n${SEP2}\n`;
  const recent = [...state.resolvedPredictions].reverse().slice(0, 20);
  for (const e of recent) {
    const ok   = e.correct ? 'ĐÚNG ✓' : 'SAI  ✗';
    const flags = [
      e.manipDetected ? '[⚠MAN]' : '',
      e.autoBroken    ? '[⚡BRK]' : ''
    ].filter(Boolean).join('');
    out += `  ${String(e.phien).padEnd(12)} ${(e.actual||'?').padEnd(8)} ${(e.finalPred||e.rawPred||'?').padEnd(10)} ${ok} ${flags}\n`;
  }

  // Mini history
  out += `${SEP}\n  20 phiên: `;
  const hist20 = (data.history || []).slice(0, 20);
  out += hist20.map(s => s.result[0] + s.point).join(' ') + '\n';
  out += `  [Poll ${POLL_INTERVAL/1000}s | V5 Elite | Tổng ${state.totalResolved} phiên học]\n${SEP}`;

  return out;
}

// ══════════════════════════════════════════════════════════════════
//  FETCH API
// ══════════════════════════════════════════════════════════════════
async function fetchData() {
  try {
    const ctrl    = new AbortController();
    const timer   = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT);
    const res     = await fetch(API_URL, {
      headers: { 'User-Agent': 'MD5PredictorV5/elite', 'Accept': 'application/json' },
      signal: ctrl.signal
    });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch(e) {
    console.error(`[Fetch] ${e.message}`);
    return null;
  }
}

// ══════════════════════════════════════════════════════════════════
//  MAIN LOOP
// ══════════════════════════════════════════════════════════════════
async function mainLoop() {
  console.log('[ MD5 Predictor V5 ELITE — Khởi động ]');
  restoreFromDisk();

  let saveCounter = 0;

  while (true) {
    const data = await fetchData();
    if (!data?.history || data.history.length < 5) { await sleep(POLL_INTERVAL); continue; }

    const history = data.history.slice(0, HISTORY_WINDOW);
    const latest  = history[0];
    const curPhien= latest.phien;

    if (curPhien !== state.lastPhien) {
      // ── BƯỚC 1: Học từ phiên vừa xong ─────────────────────────
      if (state.pendingPrediction && state.pendingDetail) {
        const actual     = latest.result;
        const rawPred    = state.pendingPrediction;
        const finalPred  = state.finalPrediction || rawPred;
        const wasCorrect = finalPred === actual;
        const breakUsed  = state.pendingBreakUsed;
        const manipBroke = state.manipBroke || false;

        // Online weight update
        weightManager.update(state.pendingDetail, rawPred, actual);

        // Ensemble update
        ensembleEng.update(state.pendingModelPreds || {}, actual);

        // AutoCorrector update
        autoCorrector.updateAfterResult(breakUsed, wasCorrect);

        // Log
        state.resolvedPredictions.push({
          phien:          curPhien,
          actual,
          rawPred,
          finalPred,
          correct:        wasCorrect,
          manipDetected:  manipBroke,
          autoBroken:     breakUsed,
          taiP:           state.pendingRawProb,
        });
        if (state.resolvedPredictions.length > MAX_TRACK)
          state.resolvedPredictions = state.resolvedPredictions.slice(-MAX_TRACK);

        state.totalResolved++;
      }

      state.lastPhien = curPhien;

      // ── BƯỚC 2: Train các models ────────────────────────────────
      markovEng.train(history);
      patternMem.train(history);
      streakAnalyzer.train(history);

      // ── BƯỚC 3: Chạy prediction pipeline ───────────────────────
      const pred = runPrediction(history);
      state.patternResult = pred.patResult;
      state.streakInfo    = { cur: pred.streakCur, len: pred.streakLen };

      // ── BƯỚC 4: Manipulation detection ─────────────────────────
      const manipReport = manipDetector.analyze(history);
      state.manipReport = manipReport;

      // ── BƯỚC 5: Auto-correction ─────────────────────────────────
      const acInfo = autoCorrector.shouldBreak(
        state.resolvedPredictions, pred.rawPred, pred.blendedProb
      );

      // ── BƯỚC 6: Manipulation counter-strategy ───────────────────
      const manipInfo = manipDetector.getCounterStrategy(
        acInfo.corrected, manipReport,
        { breakCount: autoCorrector.breakCount }
      );

      // Ưu tiên: manipulation > auto-correct
      let finalPred  = manipInfo.pred;
      let brokeFlag  = manipInfo.broke;
      let breakReason= manipInfo.broke ? `MANIP:${manipInfo.reason}` : (acInfo.break ? acInfo.reason : null);

      if (!brokeFlag && acInfo.break) {
        finalPred  = acInfo.corrected;
        brokeFlag  = true;
        breakReason= acInfo.reason;
      }

      // ── BƯỚC 7: Cập nhật state ───────────────────────────────────
      state.pendingPrediction = pred.rawPred;
      state.pendingRawProb    = pred.blendedProb;
      state.pendingDetail     = pred.features;
      state.pendingModelPreds = pred.modelProbs;
      state.pendingBreakUsed  = brokeFlag;
      state.manipBroke        = manipInfo.broke;
      state.finalPrediction   = finalPred;
      state.latestPrediction  = finalPred;
      state.finalBreakReason  = breakReason;

      // Xác suất hiển thị (đảo nếu bẻ)
      if (finalPred === 'TAI') {
        state.taiP = pred.taiP; state.xiuP = pred.xiuP;
      } else {
        state.taiP = pred.xiuP; state.xiuP = pred.taiP;
      }

      state.autoCorrectInfo = { broke: brokeFlag, reason: breakReason };

      // ── Accuracy & Learning phase ────────────────────────────────
      state.accuracy    = calcAccuracy();
      state.recentLog   = [...state.resolvedPredictions].reverse().slice(0, 15);
      state.ensembleWeights = { ...ensembleEng.modelWeights };
      state.featureWeights  = { ...weightManager.weights };
      state.featureAccuracy = { ...weightManager.accuracy };

      const total = state.totalResolved;
      if (total < COLD_START_MIN)    state.learningPhase = 'COLD_START';
      else if (total < COLD_START_MIN + 50) state.learningPhase = 'LEARNING';
      else                           state.learningPhase = 'OPTIMIZED';

      // ── Render ───────────────────────────────────────────────────
      state.latestRender = buildTerminalRender(data, pred);
      console.log(`[V5] Phiên ${curPhien} | Dự đoán: ${finalPred}${brokeFlag?' (BẺ)':''} | Acc: ${pct(state.accuracy.acc)} (${state.accuracy.total} phiên)`);

      // Save mỗi 10 phiên
      if (++saveCounter % 10 === 0) saveToDisk();
    }

    await sleep(POLL_INTERVAL);
  }
}

// ══════════════════════════════════════════════════════════════════
//  EXPRESS SERVER + DASHBOARD
// ══════════════════════════════════════════════════════════════════
const app = express();
app.use(express.static('public'));

// ── Web Dashboard (HTML đẹp, responsive) ────────────────────────
app.get('/', (req, res) => {
  const s     = state;
  const acc   = s.accuracy;
  const acc10 = calcAccuracy(10);
  const acc30 = calcAccuracy(30);
  const pred  = s.latestPrediction || '?';
  const mr    = s.manipReport;
  const ai    = s.autoCorrectInfo;
  const si    = s.streakInfo;
  const phase = s.learningPhase;
  const phaseColor = { COLD_START: '#888', LEARNING: '#fa0', OPTIMIZED: '#0f0' };

  const accBar = (v, w = 180) => {
    const n = Math.round(clamp(v, 0, 1) * w);
    const color = v >= 0.55 ? '#0f0' : v >= 0.45 ? '#fa0' : '#f44';
    return `<div style="display:inline-block;background:#222;width:${w}px;height:14px;border-radius:3px;overflow:hidden;vertical-align:middle">
      <div style="background:${color};width:${n}px;height:100%;"></div></div>`;
  };
  const miniBar = (v, w = 100) => {
    const n = Math.round(clamp(v, 0, 1) * w);
    const color = v > 0.55 ? '#4af' : v < 0.45 ? '#f84' : '#888';
    return `<div style="display:inline-block;background:#1a1a1a;width:${w}px;height:10px;border-radius:2px;overflow:hidden;vertical-align:middle">
      <div style="background:${color};width:${n}px;height:100%;"></div></div>`;
  };

  // Lịch sử gần đây
  const recentRows = [...s.resolvedPredictions].reverse().slice(0, 20).map(e => {
    const cls   = e.correct ? 'color:#0f0' : 'color:#f44';
    const ok    = e.correct ? '✓ ĐÚNG' : '✗ SAI';
    const flags = [
      e.manipDetected ? '<span style="color:#fa0">⚠MAN</span>' : '',
      e.autoBroken    ? '<span style="color:#a0f">⚡BRK</span>' : ''
    ].filter(Boolean).join(' ');
    return `<tr>
      <td>${e.phien}</td>
      <td>${e.actual}</td>
      <td>${e.finalPred||e.rawPred||'?'}</td>
      <td style="${cls};font-weight:bold">${ok}</td>
      <td>${flags}</td>
    </tr>`;
  }).join('');

  // Ensemble breakdown
  const ensembleRows = Object.entries(s.ensembleWeights||{}).map(([name, w]) => {
    const acc_  = (s.featureAccuracy ? ensembleEng.modelAccuracy[name] : 0.5) || 0.5;
    const prob  = 0.5; // placeholder
    return `<tr>
      <td>${name}</td>
      <td>${miniBar(w)} ${pct(w)}</td>
      <td>${miniBar(acc_)} ${pct(acc_)}</td>
    </tr>`;
  }).join('');

  res.send(`<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MD5 Predictor V5 Elite</title>
  <meta http-equiv="refresh" content="2">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#0a0a0a;color:#ccc;font-family:'Courier New',monospace;font-size:13px;padding:12px}
    h1{color:#0f0;font-size:16px;margin-bottom:4px}
    .badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:bold}
    .card{background:#111;border:1px solid #222;border-radius:6px;padding:12px;margin-bottom:10px}
    .card h2{font-size:12px;color:#0af;margin-bottom:8px;border-bottom:1px solid #222;padding-bottom:4px}
    .pred-box{font-size:32px;font-weight:bold;text-align:center;padding:16px;border-radius:8px;margin:8px 0}
    .pred-tai{background:linear-gradient(135deg,#001a00,#003300);color:#00ff00;border:2px solid #0f0}
    .pred-xiu{background:linear-gradient(135deg,#1a0000,#330000);color:#ff4444;border:2px solid #f44}
    .broken{color:#a0f;font-size:12px}
    table{width:100%;border-collapse:collapse;font-size:12px}
    td,th{padding:3px 6px;border-bottom:1px solid #1a1a1a;text-align:left}
    th{color:#666;font-weight:normal}
    .warn{background:#1a1000;border:1px solid #fa0;border-radius:4px;padding:8px;color:#fa0;margin-bottom:8px}
    .circuit{background:#1a0000;border:1px solid #f44;border-radius:4px;padding:8px;color:#f88;margin-bottom:8px}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    @media(max-width:600px){.grid2{grid-template-columns:1fr}}
    .phase{font-size:11px;padding:2px 8px;border-radius:3px;font-weight:bold}
    .acc-num{font-size:22px;font-weight:bold;color:#0f0}
  </style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
  <div>
    <h1>🎯 MD5 PREDICTOR V5 ELITE</h1>
    <small style="color:#555">${new Date().toLocaleString('vi-VN')} &nbsp;|&nbsp; Phiên học: ${s.totalResolved}</small>
  </div>
  <span class="phase" style="background:${phaseColor[phase]||'#888'}22;color:${phaseColor[phase]||'#888'};border:1px solid ${phaseColor[phase]||'#888'}">${phase}</span>
</div>

${mr?.isManipulated ? `<div class="warn">⚠ CẢNH BÁO NHÀCÁI CAN THIỆP — Score: ${pct(mr.score)} [${mr.type}]<br>
  ${(mr.signals||[]).map(s=>`<small>• ${s.desc}</small>`).join('<br>')}</div>` : ''}
${autoCorrector.circuitOpen ? `<div class="circuit">⛔ CIRCUIT BREAKER ĐANG MỞ — Tạm dừng auto-correction (cooldown: ${autoCorrector.circuitCooldown})</div>` : ''}

<div class="grid2">
  <!-- Dự đoán -->
  <div class="card">
    <h2>DỰ ĐOÁN PHIÊN TIẾP</h2>
    <div class="pred-box pred-${pred === 'TAI' ? 'tai' : 'xiu'}">
      ${pred}
      ${ai?.broke ? `<br><span class="broken">⚡ ĐÃ BẺ: ${ai.reason||''}</span>` : ''}
    </div>
    <table>
      <tr><td>TAI</td><td>${miniBar(s.taiP, 120)} ${pct(s.taiP)}</td></tr>
      <tr><td>XIU</td><td>${miniBar(1-s.taiP, 120)} ${pct(s.xiuP)}</td></tr>
    </table>
    <div style="margin-top:8px;color:#666;font-size:11px">
      Cầu hiện tại: <b style="color:#ccc">${si.cur||'?'}</b> × ${si.len}
    </div>
  </div>

  <!-- Accuracy -->
  <div class="card">
    <h2>THỐNG KÊ CHÍNH XÁC</h2>
    <div style="text-align:center;margin:8px 0">
      <div class="acc-num">${pct(acc.acc)}</div>
      <small style="color:#555">Tổng ${acc.total} phiên | Đúng ${acc.correct} / Sai ${acc.wrong}</small>
    </div>
    <table>
      <tr><th>Window</th><th>Accuracy</th><th></th></tr>
      <tr><td>10 phiên</td><td>${pct(acc10.acc)}</td><td>${accBar(acc10.acc, 80)}</td></tr>
      <tr><td>30 phiên</td><td>${pct(acc30.acc)}</td><td>${accBar(acc30.acc, 80)}</td></tr>
      <tr><td>Tổng cộng</td><td>${pct(acc.acc)}</td><td>${accBar(acc.acc, 80)}</td></tr>
    </table>
    <div style="margin-top:6px;font-size:11px;color:#666">
      Break count: ${Math.floor(autoCorrector.breakCount)} / ${autoCorrector.MAX_SEQ_BREAK}
    </div>
  </div>
</div>

<div class="grid2">
  <!-- Ensemble -->
  <div class="card">
    <h2>ENSEMBLE MODELS</h2>
    <table>
      <tr><th>Model</th><th>Trọng số</th><th>Accuracy</th></tr>
      ${ensembleRows}
    </table>
  </div>

  <!-- Pattern -->
  <div class="card">
    <h2>PATTERN MEMORY — ${patternMem.getTotalPatterns()} mẫu</h2>
    ${s.patternResult?.pattern ? `
    <div>Pattern: <b style="color:#af0">"${s.patternResult.pattern}"</b></div>
    <div>Dự đoán: <b>${s.patternResult.prediction}</b>  Tin cậy: ${pct(s.patternResult.confidence)}</div>
    ${s.patternResult.stats ? `<div style="color:#555;font-size:11px">TAI~${s.patternResult.stats.TAI} / XIU~${s.patternResult.stats.XIU}</div>` : ''}
    ` : '<div style="color:#555">Chưa đủ dữ liệu pattern.</div>'}
    <hr style="border-color:#1a1a1a;margin:8px 0">
    <div style="font-size:11px;color:#666">Markov: bậc 1, 2, 3 song song</div>
  </div>
</div>

<!-- Lịch sử -->
<div class="card">
  <h2>LỊCH SỬ DỰ ĐOÁN (20 phiên gần nhất)</h2>
  <table>
    <tr><th>Phiên</th><th>Thực tế</th><th>Dự đoán</th><th>Kết quả</th><th>Flags</th></tr>
    ${recentRows || '<tr><td colspan="5" style="color:#555">Chưa có dữ liệu</td></tr>'}
  </table>
</div>

<div style="color:#333;font-size:10px;text-align:center">V5 Elite | Auto-refresh 2s | Poll ${POLL_INTERVAL/1000}s</div>
</body></html>`);
});

// ── API endpoint ─────────────────────────────────────────────────
app.get('/api', (req, res) => {
  res.json({
    version:          'v5-elite',
    prediction:       state.latestPrediction,
    taiP:             state.taiP,
    xiuP:             state.xiuP,
    accuracy:         state.accuracy,
    acc10:            calcAccuracy(10),
    acc30:            calcAccuracy(30),
    manipReport:      state.manipReport,
    autoCorrect:      state.autoCorrectInfo,
    streakInfo:       state.streakInfo,
    patternResult:    state.patternResult,
    ensembleWeights:  state.ensembleWeights,
    featureWeights:   state.featureWeights,
    learningPhase:    state.learningPhase,
    totalResolved:    state.totalResolved,
    recentLog:        state.recentLog,
    circuitOpen:      autoCorrector.circuitOpen,
    breakCount:       autoCorrector.breakCount,
  });
});

// ── Terminal view (plain text) ───────────────────────────────────
app.get('/terminal', (req, res) => {
  res.type('text/plain; charset=utf-8');
  res.send(state.latestRender || 'Đang chờ dữ liệu...');
});

// ── Reset (debug) ─────────────────────────────────────────────────
app.get('/reset', (req, res) => {
  try { fs.unlinkSync(STATE_FILE); } catch(e) {}
  res.json({ ok: true, msg: 'State đã reset. Khởi động lại server.' });
});

// ── Force save ─────────────────────────────────────────────────
app.get('/save', (req, res) => {
  saveToDisk();
  res.json({ ok: true, msg: 'Đã lưu state.', totalResolved: state.totalResolved });
});

// ══════════════════════════════════════════════════════════════════
app.listen(PORT, () => {
  console.log(`[V5] Server đang chạy tại http://localhost:${PORT}`);
  mainLoop().catch(err => {
    console.error('[V5] Lỗi nghiêm trọng trong vòng lặp:', err);
    process.exit(1);
  });
});

// Graceful shutdown
process.on('SIGTERM', () => { saveToDisk(); process.exit(0); });
process.on('SIGINT',  () => { saveToDisk(); process.exit(0); });