// Xuất dashboard thành một trang tĩnh để đưa lên Vercel (hoặc bất kỳ host tĩnh nào).
//
//   node scripts/build_static_site.mjs            -> ghi ra thư mục site/
//
// Máy chủ FastAPI và pipeline YOLO không chạy được trên host tĩnh (PyTorch vượt
// xa giới hạn dung lượng của hàm serverless). Vì vậy bản trực tuyến là bản trình
// diễn: nó đọc đúng kết quả đã tính sẵn và commit trong data/outputs/, qua đúng
// giao diện dashboard, chỉ thay các endpoint /api/* bằng tệp JSON tương ứng.
// Không cần cài gói npm nào - chỉ dùng thư viện chuẩn của Node.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, process.argv[2] || 'site');
const DATA = path.join(OUT, 'data');

const read = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf-8'));
const write = (name, obj) =>
  fs.writeFileSync(path.join(DATA, name), JSON.stringify(obj));

// Đọc ngưỡng thẳng từ tệp cấu hình để trang tĩnh không lệch với pipeline.
function yamlNumber(text, key, fallback) {
  const m = text.match(new RegExp(`^\\s*${key}:\\s*([0-9.]+)`, 'm'));
  return m ? Number(m[1]) : fallback;
}
function yamlString(text, key, fallback) {
  const m = text.match(new RegExp(`^\\s*${key}:\\s*(.+)$`, 'm'));
  return m ? m[1].trim().replace(/^['"]|['"]$/g, '') : fallback;
}

fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(DATA, { recursive: true });

const results = read('data/outputs/results.json');
const cfgText = fs.readFileSync(path.join(ROOT, 'configs/synthetic.yaml'), 'utf-8');

// Video minh hoạ: đoạn cắt ngắn đã chuyển mã H.264 để mọi trình duyệt phát được.
const clip = path.join(ROOT, 'docs/assets/demo_overlay.mp4');
const hasVideo = fs.existsSync(clip);
if (hasVideo) fs.copyFileSync(clip, path.join(DATA, 'video.mp4'));

write('summary.json', {
  summary: results.summary || {},
  site: results.site || yamlString(cfgText, 'name', ''),
  site_short: yamlString(cfgText, 'short_name', ''),
  thresholds: {
    ttc: yamlNumber(cfgText, 'ttc_threshold', 3.0),
    pet: yamlNumber(cfgText, 'pet_threshold', 1.5),
    severe_ttc: 1.5,
  },
  has_video: hasVideo,
});

// Giống hệt /api/events?limit=2000: mới nhất trước.
const events = [...(results.events || [])]
  .sort((a, b) => (b.t || 0) - (a.t || 0))
  .slice(0, 2000);
write('events.json', { events, count: events.length });
write('riskmap.json', results.risk_map || { cells: [], hotspots: [], bounds: null });
write('timeline.json', { timeline: results.timeline || [] });
write('behaviors.json', { behaviors: results.behaviors || [] });

const evalPath = path.join(ROOT, 'data/outputs/evaluation.json');
write('evaluation.json', fs.existsSync(evalPath)
  ? { available: true, ...read('data/outputs/evaluation.json') }
  : { available: false });

// Trang giao diện: bật chế độ tĩnh và gắn dòng ghi chú để người xem biết đây là
// dữ liệu tính sẵn, không phải camera đang chạy.
const s = results.summary || {};
const note =
  `Bản trình diễn trực tuyến: dữ liệu tính sẵn từ video mô phỏng ` +
  `${Math.round(s.duration_s || 0)} giây (${s.n_frames || 0} khung hình). ` +
  `Pipeline đầy đủ chạy cục bộ bằng lệnh <code>saferoad serve</code>.`;
let html = fs.readFileSync(
  path.join(ROOT, 'src/saferoad/dashboard/static/index.html'), 'utf-8');
html = html.replace('<script>',
  '<script>window.SAFEROAD_STATIC = true;</script>\n<script>');
html = html.replace('<main class="main">',
  '<main class="main">\n    <div class="static-note" style="margin:0 0 12px;padding:8px 12px;' +
  'border:1px solid #334155;border-radius:8px;font-size:13px;color:#cbd5e1;' +
  `background:#0f172a">${note}</div>`);
fs.writeFileSync(path.join(OUT, 'index.html'), html);

const files = fs.readdirSync(DATA);
console.log(`✓ ${path.relative(ROOT, OUT)}/ - index.html + data/{${files.join(', ')}}`);
console.log(`  ${events.length} sự kiện · video minh hoạ: ${hasVideo ? 'có' : 'không'}`);
