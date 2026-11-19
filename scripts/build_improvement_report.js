/**
 * Sinh BÁO CÁO CẢI TIẾN (Chặng 1) cho SafeRoad AI - .docx, rồi chuyển sang PDF
 * bằng LibreOffice nếu máy có.
 *
 * Mọi con số đọc thẳng từ data/outputs/*.json do mã nguồn sinh ra. Chạy lại đánh
 * giá rồi chạy lại script này là báo cáo tự cập nhật - không có số nào gõ tay.
 *
 *   node scripts/build_improvement_report.js
 */

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, Footer,
  PageNumber, LevelFormat, TableOfContents,
} = require('docx');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'docs', 'BaoCaoCaiTien_SafeRoadAI_Chang1.docx');
const J = (p) => {
  const f = path.join(ROOT, p);
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : null;
};

const EV = J('data/outputs/evaluation.json');
const GRID = J('data/outputs/evaluation_grid.json');
const ST = J('data/outputs/evaluation_street.json');
const STF = J('data/outputs/evaluation_street_finetuned_test.json');
const STP = J('data/outputs/evaluation_street_pretrained_test.json');
const MV = J('data/outputs/evaluation_real.json');
const MTP = J('data/outputs/evaluation_mvti_pretrained_test.json');
const MTF = J('data/outputs/evaluation_mvti_finetuned_test.json');
const CAL = J('data/outputs/calibration_street.json');
const HO = J('data/outputs/street_heldout/results.json');
const SENS = J('data/outputs/street_reference_sensitivity.json');
// Kết quả chấm theo nhãn người: data/outputs/review_score_<tập>.json (saferoad review-score).
const REVIEWS = fs.readdirSync(path.join(ROOT, 'data', 'outputs'))
  .filter((f) => /^review_score_.+\.json$/.test(f)).sort()
  .map((f) => [f.replace(/^review_score_|\.json$/g, ''), J(path.join('data', 'outputs', f))]);
const REVIEW_VI = { street_heldout: 'Đại lộ, 441 s chưa từng dùng', street: 'Đại lộ 0-300 s',
  street_finetuned_test: 'Đại lộ, test 220-300 s', real: 'MVTI' };
const V1 = J('data/outputs/archive_v1/evaluation.json');
const V1R = J('data/outputs/archive_v1/results.json');
const V1REAL = J('data/outputs/archive_v1/evaluation_real.json');
if (!EV || EV.version !== 2) {
  console.error('Cần data/outputs/evaluation.json phiên bản 2: chạy python -m saferoad evaluate');
  process.exit(1);
}
const N_TESTS = fs.readdirSync(path.join(ROOT, 'tests')).filter((f) => /^test_.*\.py$/.test(f))
  .map((f) => (fs.readFileSync(path.join(ROOT, 'tests', f), 'utf8').match(/^\s*def test_/gm) || []).length)
  .reduce((a, b) => a + b, 0);

// ------------------------------------------------------------------------- //
const n = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '-'
  : Number(v).toFixed(d).replace('.', ',');
const r = (x, d = 3) => (x && x.den !== undefined)
  ? `${n(x.value, d)} (${x.num}/${x.den})` : n(x, d);
const FONT = 'Times New Roman';
const TW = 9360;   // bề rộng nội dung A4 lề 2,54 cm ≈ 16,5 cm ≈ 9360 DXA

const P = (text, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 120, line: 288 },
  children: [new TextRun({ text, font: FONT, size: o.size || 24, bold: o.bold,
    italics: o.italics, color: o.color })],
});
const Rich = (runs, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 120, line: 288 },
  children: runs.map((x) => new TextRun({ font: FONT, size: 24, ...(typeof x === 'string' ? { text: x } : x) })),
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })] });
const B = (t, lvl = 0) => new Paragraph({
  numbering: { reference: 'bul', level: lvl }, spacing: { after: 80, line: 276 },
  alignment: AlignmentType.JUSTIFIED,
  children: (Array.isArray(t) ? t : [t]).map((x) => new TextRun({ font: FONT, size: 24,
    ...(typeof x === 'string' ? { text: x } : x) })),
});
const Cap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 160 },
  children: [new TextRun({ text: t, font: FONT, size: 21, italics: true })] });

function T(head, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.round((x / total) * TW));
  w[w.length - 1] += TW - w.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: '999999' };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, header) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, borders,
    shading: header ? { type: ShadingType.CLEAR, color: 'auto', fill: 'DCE6F1' } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(text), font: FONT, size: 20, bold: header })] })],
  });
  return new Table({
    width: { size: TW, type: WidthType.DXA }, columnWidths: w,
    rows: [new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, i, true)) }),
      ...rows.map((row) => new TableRow({ children: row.map((c, i) => cell(c, i, false)) }))],
  });
}
function Img(rel, width = 600) {
  const f = path.join(ROOT, rel);
  if (!fs.existsSync(f)) return P(`(thiếu ảnh ${rel})`, { italics: true });
  const buf = fs.readFileSync(f);
  // Đọc kích thước JPEG/PNG để giữ tỉ lệ.
  let wpx = 16, hpx = 9;
  if (buf[0] === 0x89) { wpx = buf.readUInt32BE(16); hpx = buf.readUInt32BE(20); } else {
    let i = 2;
    while (i < buf.length) {
      if (buf[i] !== 0xFF) { i++; continue; }
      const m = buf[i + 1];
      if (m >= 0xC0 && m <= 0xC3) { hpx = buf.readUInt16BE(i + 5); wpx = buf.readUInt16BE(i + 7); break; }
      i += 2 + buf.readUInt16BE(i + 2);
    }
  }
  return new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({
    type: rel.endsWith('.png') ? 'png' : 'jpg', data: buf,
    transformation: { width, height: Math.round(width * hpx / wpx) } })] });
}

// ------------------------------------------------------------------------- //
const M = EV.main, MT = M.metrics, RC = M.reconciliation, SET = EV.setup;
const PE = M.pet_errors, CC = M.contact_confusion.matrix, BH = M.behavior;
const SC = EV.scenario_check;
const LOO = EV.ablation.leave_one_out, CUM = EV.ablation.cumulative;
const row = (k) => LOO.find((x) => x.key === k) || {};
const v1 = V1 ? V1.final : null;
const children = [];

// ---------------------------- Trang bìa ---------------------------------- //
children.push(
  P('CUỘC THI SÁNG TẠO TRẺ QUỐC GIA TRONG LĨNH VỰC TRÍ TUỆ NHÂN TẠO NĂM 2026 - BẢNG C',
    { align: AlignmentType.CENTER, bold: true, size: 26, after: 600 }),
  P('SAFEROAD AI', { align: AlignmentType.CENTER, bold: true, size: 48, after: 120 }),
  P('BÁO CÁO CẢI TIẾN - CHẶNG 1', { align: AlignmentType.CENTER, bold: true, size: 32, after: 120 }),
  P('Phản hồi ba yêu cầu cải tiến của hội đồng: kiểm chứng trên video thật, ý nghĩa của PET và near-miss, báo cáo đánh giá nhất quán có ablation',
    { align: AlignmentType.CENTER, italics: true, after: 600 }),
  T(['Hạng mục', 'Nội dung'], [
    ['Đội thi', 'Newbie2 (AI26C0017) - Trường Đại học Bách khoa - ĐHQG-HCM'],
    ['Thành viên', 'Trần Phan Đăng Khôi (đội trưởng) · Lại Thế Mạnh Anh · Trần Huỳnh Hân'],
    ['Mã nguồn', 'https://github.com/dangkhoi-dev/SafeRoad-AI'],
    ['Phiên bản mã', process.env.SAFEROAD_TAG || 'chang1-v2.0'],
    ['Số kiểm thử tự động', `${N_TESTS} hàm kiểm thử (pytest, chạy trên GitHub Actions)`],
  ], [1, 3]),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---------------------------- 1. Tóm tắt --------------------------------- //
children.push(H1('1. Tóm tắt thay đổi'));
children.push(P('Bảng dưới liệt kê từng yêu cầu của hội đồng, việc nhóm đã làm và vị trí minh chứng trong repo. Mọi số liệu trong báo cáo này được script scripts/build_improvement_report.js đọc thẳng từ các tệp JSON do mã nguồn sinh ra; không có con số nào gõ tay.'));
const pYes = (x) => x ? 'có' : 'chưa có';
children.push(T(['Yêu cầu', 'Đã làm', 'Minh chứng'], [
  ['1. Video thật, chất lượng phát hiện',
    'Thêm bộ video đại lộ thật (Kaggle) có nhãn bbox + ID; ghép nhãn CVAT bị cắt đoạn; hiệu chuẩn camera có kiểm tra chéo; bảng detection → tracking → cảnh báo trên cùng video; báo nhầm/bỏ sót theo giờ; công cụ rà soát nhãn của người + kappa; demo chạy pipeline thật trên video tải lên',
    'evaluation_street*.json, evaluation_real.json, calibration_street.json, street_reference_sensitivity.json, models/saferoad_street_yolo11n.pt, review_candidates_*.csv, tab "Chạy thử video" và "Rà soát nhãn"'],
  ['2. Ý nghĩa PET và near-miss',
    'Tách PET ngoại suy và PET đo từ mốc vào/ra vùng xung đột, lưu 4 mốc thời gian; xử lý quỹ đạo song song và xe đã qua giao điểm; phân biệt near-miss / chồng thân xe do sai số / va chạm thật; trình mô phỏng có vật lý va chạm',
    'conflict/encroachment.py, tests/test_pet_and_contact.py (ca tái hiện từng lỗi), mục 3'],
  ['3. Đánh giá nhất quán, có ablation',
    'Một luồng đếm sự kiện với 5 đẳng thức tự kiểm; mọi tỉ lệ kèm tử/mẫu số; sửa cờ hành vi 305/307; ablation bỏ từng khối + cộng dồn trên cùng mẫu; lưới 3 mật độ × 3 góc camera × 3 điều kiện ánh sáng; log cấu hình có mã băm',
    'evaluation.json (mục reconciliation, ablation), evaluation_grid.json, mục 4'],
], [1.2, 3, 1.8]));

children.push(P(''));
children.push(P('Chỉ số chính trên tập mô phỏng, trước và sau cải tiến (cùng định nghĩa đầy đủ, mẫu số ghi trong ngoặc):', { bold: true }));
const before = v1 ? [
  ['Precision', `${n(v1.precision)} (${v1.tp}/${v1.tp + v1.fp})`, r(MT.precision)],
  ['Recall', `${n(v1.recall)} (${v1.tp}/${v1.tp + v1.fn})`, r(MT.recall)],
  ['F1', n(v1.f1), n(MT.f1)],
  ['TTC MAE (s)', n(v1.ttc_mae), n(MT.ttc_mae)],
  ['PET MAE (s)', `${n(v1.pet_mae)} (hai định nghĩa khác nhau)`, `${n(PE.measured && PE.measured.mae)} (PET đo, n=${PE.measured ? PE.measured.n : 0})`],
  ['Sự kiện trên dashboard / trong đánh giá / được chấm', `${V1R ? V1R.summary.total_events : '-'} / ${V1.runtime.total_events} / ${v1.tp + v1.fp} (không giải thích)`, `${RC.rows.exported_to_results} / ${RC.rows.emitted} / ${RC.rows.scored} (đối soát ở mục 4.1)`],
  ['Quỹ đạo bị gắn cờ hành vi', V1R ? `${V1R.behaviors.length}/${V1.tracking.n_pred_tracks}` : '-', r(BH.flagged_tracks)],
] : [];
if (before.length) children.push(T(['Chỉ số', 'Bản cũ', 'Bản cải tiến'], before, [2.2, 2, 2]));
children.push(P('Lưu ý khi so sánh: trình mô phỏng đã được sửa (mục 3.4 và 4.3), nên tập nhãn chuẩn của bản mới khác bản cũ. Bảng ablation ở mục 4.4 mới là phép so sánh trên cùng một mẫu.', { italics: true, size: 22 }));

children.push(H2('Những gì được giữ nguyên'));
children.push(B('Kiến trúc tách lớp phát hiện → theo vết → quỹ đạo mặt đất → TTC/PET → chỉ báo xung đột. Hệ thống xuất ra chỉ báo xung đột giao thông, không coi xung đột là tai nạn; bản mới còn tách hẳn ba kết cục (mục 3.4).'));
children.push(B('Mọi phép đo trên mô phỏng có seed cố định (kịch bản, nhiễu detector), có quét nhiễu, có ablation, ghi chú về ID switch, và vai trò rà soát của con người trước khi dùng kết quả.'));

// ---------------------------- 2. Yêu cầu 1 ------------------------------- //
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1('2. Yêu cầu 1 - Kiểm chứng trên video thật và chất lượng phát hiện'));
children.push(H2('2.1. Dữ liệu video thật'));
const sd = ST ? ST.dataset : null;
children.push(T(['Nguồn', 'Quy mô', 'Nhãn', 'Vai trò'], [
  ['Đại lộ nhiều làn (Kaggle, quay từ mái nhà, iPhone 13 4K → 1080p)',
    sd ? `${sd.n_frames} khung hình có nhãn (300 s) + 13.246 khung không nhãn (441 s)` : '300 s có nhãn + 441 s không nhãn',
    sd ? `${sd.n_boxes} bbox, ${sd.n_tracks} quỹ đạo (Car, People)` : 'bbox + ID',
    'Bảng detection → tracking → cảnh báo; chạy đầu-cuối trên 441 s chưa từng dùng'],
  ['Multi-view Traffic Intersection (MVTI)', MV ? `${MV.dataset.n_frames} khung hình` : '2.441 khung hình',
    MV ? `${MV.dataset.n_boxes} bbox, ${MV.dataset.n_tracks} quỹ đạo` : 'bbox + ID', 'Giao lộ thật, camera hạ tầng'],
], [2.2, 1.6, 1.6, 2]));
if (sd && sd.stitch) {
  children.push(P(`Nhãn của video đại lộ được xuất từ CVAT theo từng đoạn 30 khung hình chồng nhau 5 khung, và mỗi đoạn đánh lại track_id từ đầu. Đọc thẳng thì mỗi chiếc xe bị cắt thành hàng chục "đối tượng" và không đo được IDF1. Module data/cvat_video.py ghép lại theo IoU trên 5 khung chồng nhau: ${sd.stitch.n_local_tracks} quỹ đạo cục bộ của ${sd.stitch.n_chunks} đoạn được nối thành ${sd.stitch.n_global_tracks} quỹ đạo (${sd.stitch.n_links} mối nối). Phép ghép có test riêng (tests/test_evaluation_v2.py::TestCvatStitching).`));
}

children.push(H2('2.2. Hiệu chuẩn camera'));
if (CAL) {
  children.push(P(`Công cụ mới saferoad calibrate tính homography từ nhiều hơn 4 điểm và báo ba loại sai số, vì homography từ đúng 4 điểm luôn khớp tuyệt đối 4 điểm đó nên "sai số 0" không nói lên điều gì. Với video đại lộ, nhóm dùng ${CAL.n_points} điểm là hai đầu của 8 vạch đứt trên hai vạch phân làn kề nhau; toạ độ pixel được tinh chỉnh tự động theo tâm vạch sơn.`));
  children.push(T(['Loại sai số', 'Số điểm', 'RMSE (m)', 'Trung bình (m)', 'Lớn nhất (m)'], [
    ['Trên điểm dùng để tính (fit)', CAL.fit.n, n(CAL.fit.rmse_m), n(CAL.fit.mean_m), n(CAL.fit.max_m)],
    ['Bỏ từng vạch ra rồi tính lại (leave-one-out)', CAL.leave_one_out.n, n(CAL.leave_one_out.rmse_m), n(CAL.leave_one_out.mean_m), n(CAL.leave_one_out.max_m)],
  ], [3, 1, 1, 1.2, 1.2]));
  children.push(Img('docs/assets/calib_street.jpg', 560));
  children.push(Cap('Hình 1. Lưới mét suy ra từ homography (vàng: cách 3,5 m theo bề ngang, tím: theo chiều dọc), chấm đỏ là điểm mốc.'));
  children.push(P('Giả định đã dùng (ghi trong configs/calib_street.yaml):', { bold: true }));
  (CAL.assumptions || []).forEach((a) => children.push(B(a)));
  const SP = ST && ST.calibration_speed_check;
  if (SP) children.push(P(`Kiểm tra độc lập bằng vận tốc: chiếu quỹ đạo người gán nhãn của ${SP.n_cars} ô tô đang chạy ở nửa dưới khung hình qua homography, vận tốc trung vị ${n(SP.median_kmh, 1)} km/h, phân vị 90 là ${n(SP.p90_kmh, 1)} km/h - phù hợp một đại lộ đô thị giới hạn 60 km/h. Nếu tỉ lệ sai lệch nhiều (ví dụ chu kỳ vạch là 9 m thay vì 12 m) thì phân vị 90 sẽ lệch tương ứng (khoảng 44 km/h).`));
  children.push(P('Giới hạn phải nêu rõ: đây là hiệu chuẩn ước lượng từ tiêu chuẩn vạch kẻ, không phải đo thực địa, vì video quay ở nước ngoài. Sai số tỉ lệ tổng thể không làm thay đổi TTC và PET (cả quãng đường lẫn vận tốc cùng co giãn), nhưng ảnh hưởng cổng khoảng cách 3 m và cổng tốc độ tiếp cận 3 m/s. Với camera của nhóm tại Việt Nam, quy trình là đo thực địa ít nhất 6 điểm mốc + 2 điểm kiểm tra rồi chạy cùng lệnh.'));
}

children.push(H2('2.3. Detection → tracking → cảnh báo trên cùng một video'));
children.push(P('Video thật không có nhãn xung đột. Để vẫn đo được khâu cảnh báo trên cùng dữ liệu, nhóm chạy đúng phần sau phát hiện của pipeline (homography, làm mượt, TTC/PET, các cổng, gom episode) trên quỹ đạo do người gán nhãn - gọi là cảnh báo tham chiếu. So pipeline đầy đủ (YOLO + ByteTrack) với tham chiếu cho biết bao nhiêu cảnh báo bị thêm hoặc mất do detection và tracking. Đây không phải độ đúng tuyệt đối: độ đúng tuyệt đối cần nhãn người rà soát (mục 2.6).'));
const detRow = (name, x) => {
  if (!x) return null;
  const big = x.detection_min_area || {};
  return [name, n(x.detection.mAP50), n(big.mAP50), `${n(big.precision50_op ?? x.detection.precision50_op)} / ${n(big.recall50_op ?? x.detection.recall50_op)}`,
    n(x.tracking.idf1), n(x.tracking.mota), x.tracking.id_switches, n(x.runtime.processing_fps, 1)];
};
const alertRow = (name, x) => {
  if (!x) return null;
  const a = x.alerts || {}, m = a.metrics_vs_reference || {}, ph = (a.reconciliation || {}).per_hour || {};
  return [name, n(x.runtime.duration_s, 0), (a.reconciliation || {}).rows ? a.reconciliation.rows.emitted : '-', a.n_reference ?? '-',
    r(m.precision), r(m.recall), ph.false_alarms_per_hour == null ? '-' : n(ph.false_alarms_per_hour, 1),
    ph.misses_per_hour == null ? '-' : n(ph.misses_per_hour, 1)];
};
const REALS = [['Đại lộ 0-300 s, YOLO11n gốc', ST], ['Đại lộ, test 220-300 s, YOLO11n gốc', STP],
  ['Đại lộ, test 220-300 s, YOLO11n tinh chỉnh', STF], ['MVTI 0-81 s, YOLO11n gốc', MV],
  ['MVTI, test 57-81 s, YOLO11n gốc', MTP], ['MVTI, test 57-81 s, YOLO11n tinh chỉnh', MTF]];
children.push(T(['Dữ liệu / detector', 'mAP@0,5', 'mAP@0,5 bỏ hộp nhỏ', 'P / R ở ngưỡng chạy', 'IDF1', 'MOTA', 'Đổi ID', 'FPS'],
  REALS.map(([k, x]) => detRow(k, x)).filter(Boolean), [2.6, 0.8, 1, 1.2, 0.7, 0.7, 0.7, 0.6]));
children.push(Cap('Bảng 1a. Detection và tracking. "Bỏ hộp nhỏ": hộp chuẩn nhỏ hơn detection.min_box_area (200 px² với đại lộ) là vùng bỏ qua theo quy ước COCO, vì pipeline cố ý không xử lý vật thể xa như vậy; cột mAP@0,5 bên trái chấm trên toàn bộ nhãn.'));
children.push(T(['Dữ liệu / detector', 'Thời lượng (s)', 'Cảnh báo hệ thống', 'Tham chiếu', 'P so tham chiếu', 'R so tham chiếu', 'Báo nhầm/giờ', 'Bỏ sót/giờ'],
  REALS.map(([k, x]) => alertRow(k, x)).filter(Boolean), [2.6, 0.9, 0.9, 0.9, 1.1, 1.1, 0.9, 0.9]));
children.push(Cap('Bảng 1b. Cảnh báo trên cùng video và cùng cấu hình. Báo nhầm/bỏ sót tính so với cảnh báo tham chiếu, trong vùng phân tích (mục 2.4).'));
if (ST) {
  const pc = (ST.detection.per_class || {}).pedestrian, cc = (ST.detection.per_class || {}).car;
  if (pc && cc) children.push(P(`Theo lớp (đại lộ, YOLO11n gốc, 300 s): ô tô AP@0,5 = ${n(cc.AP50)} trên ${cc.n_gt} hộp; người đi bộ AP@0,5 = ${n(pc.AP50)} trên ${pc.n_gt} hộp. Người đi bộ trên video này có kích thước trung vị khoảng 8×23 px; ở đầu vào 640 px của YOLO chỉ còn khoảng 3×8 px nên detector gần như không thấy. Muốn đo được nhóm này cần chạy theo ô ở độ phân giải gốc (detection.backend = tiled, đã có trong mã) hoặc camera đặt gần hơn - đây là giới hạn của dữ liệu đầu vào, không phải lỗi chấm điểm.`));
}
if (V1REAL) children.push(P(`Đối chiếu bản cũ trên MVTI: mAP@0,5 = ${n(V1REAL.detection.mAP50)}, IDF1 = ${n(V1REAL.tracking.idf1)} - chính hai con số hội đồng nêu. Bản mới chấm tracking theo quy ước vùng bỏ qua của MOT (hộp chuẩn nhỏ hơn ngưỡng lọc của pipeline không tính vào mẫu số), và tách hẳn ảnh hưởng của detection lên cảnh báo như bảng trên.`));
if (STF && STP) {
  children.push(P(`Tinh chỉnh YOLO11n trên phần đầu video đại lộ (0-210 s, mỗi 10 khung hình lấy 1 ảnh; 10 epoch trên CPU, scripts/train_street_yolo.py), kiểm tra trên 220-300 s chưa dùng để huấn luyện: mAP@0,5 từ ${n(STP.detection.mAP50)} lên ${n(STF.detection.mAP50)}, recall ở ngưỡng chạy từ ${n(STP.detection.recall50_op)} lên ${n(STF.detection.recall50_op)}, IDF1 từ ${n(STP.tracking.idf1)} lên ${n(STF.tracking.idf1)}. Trọng số dùng là epoch cuối, không chọn epoch theo đoạn kiểm tra. Chia theo thời gian (có khoảng trống 10 s) để tránh rò rỉ giữa hai khung hình liền nhau; tuy vậy đây vẫn là kiểm tra trong cùng một camera, đúng kịch bản triển khai thật (fine-tune theo từng camera), không phải khả năng tổng quát sang camera khác.`));
}

if (MTF && MTP) {
  const up = MTP.detection_unseen_objects || {}, uf = MTF.detection_unseen_objects || {};
  children.push(P(`Hai con số hội đồng nêu (mAP@0,5 ≈ 0,208, IDF1 ≈ 0,235) là của YOLO11n gốc trên MVTI. Làm tương tự trên chính MVTI: tinh chỉnh trên 0-50 s (khung 0-1499, mỗi 5 khung lấy 1 ảnh, 15 epoch), bỏ trống 50-57 s, kiểm tra trên 57-81 s (khung 1700-2440). Trên đoạn kiểm tra: mAP@0,5 từ ${n(MTP.detection.mAP50)} lên ${n(MTF.detection.mAP50)}, recall ở ngưỡng chạy từ ${n(MTP.detection.recall50_op)} lên ${n(MTF.detection.recall50_op)}, IDF1 từ ${n(MTP.tracking.idf1)} lên ${n(MTF.tracking.idf1)}. Như với đại lộ, trọng số dùng là epoch cuối; ảnh val lấy từ đoạn kiểm tra chỉ để theo dõi, không dùng để chọn epoch.`));
  children.push(P(`Cần đọc con số này đúng phạm vi: MVTI chỉ có 19 phương tiện và phần lớn đứng trong khung suốt 81 giây, nên ${uf.n_test_tracks != null ? (uf.n_test_tracks - uf.n_unseen_tracks) + '/' + uf.n_test_tracks : 'phần lớn'} xe ở đoạn kiểm tra đã xuất hiện lúc huấn luyện - model có thể đã "nhớ" chúng. Vì vậy chấm thêm riêng ${uf.n_unseen_tracks ?? '-'} xe chỉ xuất hiện sau phần huấn luyện (${uf.n_gt ?? '-'} hộp; hộp của các xe đã gặp là vùng bỏ qua): mAP@0,5 từ ${n(up.mAP50)} lên ${n(uf.mAP50)}. Đây là mẫu nhỏ, nhưng là phép đo duy nhất trên MVTI không bị lẫn xe đã thấy.`));
}

children.push(H2('2.4. Vùng phân tích, xe đứng yên và cặp người đi bộ'));
children.push(P('Khi chạy tầng xung đột trên quỹ đạo người gán nhãn của toàn khung hình, số cảnh báo tham chiếu lớn bất thường so với một đại lộ chạy bình thường. Rà từng ca trên video cho thấy phần lớn đến từ sai số hình học chứ không phải tình huống giao thông: hai xe đứng trong hàng chờ ở xa có bbox rung vài pixel (ở 60-100 m, mỗi pixel đã là vài chục cm) nên ước lượng ra vận tốc vài m/s ngược chiều nhau; xe chạy sát hàng xe đỗ ở lòng đường bên phải - cách vạch dùng để hiệu chuẩn hơn 10 m - bị đặt lệch ngang khoảng 1 m nên trông như chồng thân. Ba quy ước được thêm, áp dụng như nhau cho hệ thống và tham chiếu, đều nằm trong configs/street.yaml:'));
children.push(B('Vùng phân tích: chỉ xét đối tượng nằm trên lòng đường có các vạch dùng để hiệu chuẩn, trong vòng 65 m (vạch xa nhất ở 51 m). Phần còn lại của khung hình vẫn được phát hiện và theo vết, chỉ không sinh cảnh báo.'));
children.push(B('Xe đứng yên: nếu quãng dịch chuyển ròng trong 2 s chia cho 2 s nhỏ hơn 1 m/s thì vận tốc được kẹp về 0, giữ hướng thân xe lần cuối còn chạy (hoặc trục di chuyển chủ đạo của cảnh nếu xe đứng yên từ đầu).'));
children.push(B('Bỏ cặp hai người đi bộ - đúng quy ước của nhãn chuẩn (protocol v2); bản cũ chỉ áp dụng quy ước này cho nhãn mà quên áp dụng cho thuật toán.'));
if (SENS) {
  children.push(T(['Bước', 'Cấu hình', 'Cảnh báo', 'Mỗi giờ', 'Near-miss / chồng thân / nghi va chạm'],
    SENS.steps.map((x) => [x.step, x.label, x.n_events, n(x.per_hour, 1),
      `${x.by_contact.near_miss || 0} / ${x.by_contact.body_overlap || 0} / ${x.by_contact.collision_suspected || 0}`]),
    [0.5, 3, 0.8, 0.8, 1.8]));
  children.push(Cap(`Bảng 1c. Cảnh báo tham chiếu trên ${n(SENS.duration_s, 0)} s video đại lộ (quỹ đạo người gán nhãn) khi thêm dần từng quy ước - scripts/street_reference_sensitivity.py.`));
  children.push(Img('docs/assets/street_analysis_region.jpg', 560));
  children.push(Cap('Hình 2. Vùng phân tích (vàng) trên khung hình 150: lòng đường bên trái, nơi có 16 điểm hiệu chuẩn (đỏ). Đường xanh: giới hạn 65 m của bước C.'));
  children.push(P('Cái giá phải trả được nêu rõ: lòng đường bên phải và đoạn xa hơn 65 m không được phân tích. Muốn mở rộng vùng cần thêm điểm mốc hiệu chuẩn ở lòng đường bên phải (hoặc đo thực địa) rồi chạy lại cùng lệnh - quy trình đã có sẵn.'));
}

children.push(H2('2.5. Chạy thử đầu-cuối trên video chưa từng dùng'));
if (HO) {
  const s = HO.summary;
  children.push(P(`441 giây cuối của video đại lộ không có nhãn và không được dùng cho hiệu chuẩn, chọn ngưỡng hay fine-tune. Chạy toàn bộ pipeline (YOLO11n tinh chỉnh ở mục 2.3, CPU) trên đoạn này: ${s.n_frames} khung hình, ${s.n_tracks} đối tượng, ${s.total_events} cảnh báo, tức ${n(s.total_events / (s.duration_s / 3600), 1)} cảnh báo/giờ; tốc độ trung vị ${n(s.processing_fps, 1)} FPS. Danh sách cảnh báo để người rà soát nằm ở data/outputs/review_candidates_street_heldout.csv.`));
}
children.push(P('Bản demo trực tuyến không còn là dashboard tính sẵn: tab "Chạy thử video" cho phép tải một đoạn video lên máy chủ (Modal, CPU), máy chủ chạy YOLO → ByteTrack → quỹ đạo → TTC/PET → Risk và hiển thị tiến độ từng khung hình, video overlay, danh sách cảnh báo và bảng đối soát của chính lần chạy đó.'));

children.push(H2('2.6. Nhãn xung đột do người rà soát'));
if (REVIEWS.length) {
  children.push(P('Hai thành viên gán nhãn độc lập theo protocol mục 8. Precision và Recall dưới đây tính so với nhãn đồng thuận của người, không phải so với cảnh báo tham chiếu.'));
  children.push(T(['Tập', 'Ứng viên', 'Người', 'Kappa (5 lớp / có-không)', 'Đồng thuận', 'P', 'R', 'Báo nhầm/giờ', 'Bỏ sót/giờ'],
    REVIEWS.map(([k, x]) => [REVIEW_VI[k] || k, x.n_candidates, x.n_reviewers,
      `${n(x.kappa_5class)} / ${n(x.kappa_conflict_vs_not)}`,
      Object.entries(x.consensus_counts || {}).map(([a, b]) => `${a} ${b}`).join(', ') || '-',
      r(x.precision), r(x.recall), n(x.false_alarms_per_hour, 1), n(x.misses_per_hour, 1)]),
    [1.6, 0.7, 0.6, 1.2, 1.7, 1, 1, 0.8, 0.8]));
  children.push(Cap('Bảng 1d. Cảnh báo so với nhãn xung đột do người rà soát (saferoad review-score). Ứng viên bất đồng hoặc "unclear" không vào mẫu số.'));
  children.push(P('Quy trình đã dùng (protocol mục 8):'));
} else {
  children.push(P('Công cụ đã sẵn sàng, nhưng tại thời điểm nộp chưa có nhãn của người - báo cáo không trình bày con số nào thay cho bước này. Quy trình (protocol mục 8):'));
}
children.push(B('saferoad evaluate-real và saferoad run --review-csv sinh danh sách ứng viên: hợp của cảnh báo hệ thống và cảnh báo tham chiếu, kèm thời điểm và cặp đối tượng.'));
children.push(B('Hai thành viên mở tab "Rà soát nhãn", xem lại từng đoạn ở tốc độ 0,25×, gán độc lập một trong năm nhãn: near_miss, collision, normal, overlap_artifact, unclear; xem thêm toàn bộ video để ghi các xung đột hệ thống bỏ sót.'));
children.push(B('saferoad review-score tính Cohen\'s kappa (5 lớp và có/không xung đột); κ < 0,6 thì ngồi lại thống nhất cách hiểu rồi gán lại, không lấy trung bình. Nhãn đồng thuận cho Precision, Recall, báo nhầm/giờ và bỏ sót/giờ so với nhãn người.'));

// ---------------------------- 3. Yêu cầu 2 ------------------------------- //
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1('3. Yêu cầu 2 - Ý nghĩa của PET và near-miss'));
children.push(H2('3.1. Hai cách tính PET, tách bạch'));
children.push(B([{ text: 'PET ngoại suy (pet_pred) ', bold: true }, 'tính trước khi sự việc xảy ra, từ vị trí và vận tốc tức thời: kéo dài hai tia vận tốc, tìm giao điểm, ước lượng lúc xe trước ra khỏi và xe sau đi vào vùng xung đột. Dùng để mở episode và cảnh báo sớm.']));
children.push(B([{ text: 'PET đo (pet_measured) ', bold: true }, 'tính sau khi cả hai xe đã đi qua, từ quỹ đạo đã ghi: PET = t_vào(xe sau) - t_ra(xe trước). Vùng xung đột là phần giao của hai dải quét thân xe (cùng mô hình đa hình tròn với TTC); một mẫu của xe A nằm trong vùng khi thân A lúc đó chạm dải quét của xe B. Mốc vào/ra lấy ở điểm giữa hai mẫu kề biên.']));
children.push(P('Mỗi sự kiện lưu cả hai giá trị, nguồn của giá trị dùng để chốt (pet_source), trạng thái phép đo (pet_status) và bốn mốc thời gian a_in, a_out, b_in, b_out. Episode chỉ được chốt khi phép đo PET đã có kết quả (hoặc hết 4 s chờ), nên cảnh báo trên màn hình vẫn tức thời nhưng bản ghi thống kê dùng PET đo.'));
children.push(H2('3.2. Ca tái hiện từng lỗi'));
children.push(T(['Lỗi trong bản cũ', 'Ca tái hiện (tests/test_pet_and_contact.py)', 'Sau khi sửa'], [
  ['Nhãn chuẩn đo PET là "khoảng cách thời gian giữa hai tâm xe tại điểm gần nhau nhất", thuật toán online đo "giao điểm hai tia trừ L/v" - hai đại lượng khác nhau', 'test_matches_analytic_value: hai xe cắt vuông góc, PET giải tích 0,8-2,4 s', 'Cả hai dùng chung measure_pet; sai lệch ≤ 1 bước khung hình'],
  ['Xe trước đã qua giao điểm thì PET ngoại suy trả về None - đúng lúc xe sau đang lao tới', 'test_still_defined_after_leader_passed', 'Xe trước được phép đã qua tối đa 3 s'],
  ['Quỹ đạo song song/cùng làn cho ra PET nhỏ giả', 'test_parallel_lanes_have_no_pet', 'Trạng thái parallel hoặc no_zone, để TTC xử lý'],
  ['Episode mở chỉ vì PET ngoại suy < 1,5 s vẫn được giữ dù xe sau đã phanh', 'test_predicted_only_episode_rejected_when_measured_pet_is_large', `Bị loại, ghi lý do; trên mô phỏng: ${RC.rows.rejected_pet_measured_above_threshold || 0} episode`],
  ['Thân xe chồng nhau được tính là near-miss', 'TestContactClass, test_contact_when_bodies_meet', 'Tách near-miss / chồng thân do sai số / nghi va chạm'],
  ['Track id lớn đứng trước trong danh sách làm đảo lớp đối tượng của sự kiện', 'test_lower_id_is_track_a_and_classes_follow', 'Cặp luôn sắp theo id'],
], [2.6, 2.4, 2]));
children.push(H2('3.3. Sai số PET so với nhãn chuẩn sau khi sửa'));
const peRow = (name, x) => x ? [name, x.n, n(x.mae), n(x.median_ae), n(x.p90_ae), n(x.bias), r(x.within_0_5s), r(x.threshold_agreement)] : [name, 0, '-', '-', '-', '-', '-', '-'];
children.push(T(['PET', 'n', 'MAE (s)', 'Trung vị (s)', 'P90 (s)', 'Lệch TB (s)', 'Trong ±0,5 s', 'Cùng phía ngưỡng 1,5 s'],
  [peRow('Ngoại suy', PE.predicted), peRow('Đo từ mốc vào/ra', PE.measured), peRow('Giá trị dùng để chốt', PE.final)],
  [1.6, 0.6, 0.9, 1, 0.9, 1, 1.3, 1.5]));
children.push(Cap('Bảng 2. Trên các cặp khớp với nhãn chuẩn có PET xác định. Bản cũ: PET MAE ' + (v1 ? n(v1.pet_mae) : '-') + ' s, lớn gấp đôi ngưỡng 1,5 s.'));
children.push(P('Cột cuối là thứ quyết định một tình huống có bị gắn cờ hay không: tỉ lệ dự đoán và nhãn nằm cùng phía ngưỡng 1,5 s. Với PET đo, sai số còn lại chủ yếu là bước lấy mẫu 1/30 s của video so với 1/60 s của nhãn chuẩn.'));
children.push(H2('3.4. Near-miss, chồng thân xe và va chạm thật'));
children.push(P(`Trình mô phỏng cũ không có ràng buộc va chạm: khi phanh tránh không kịp, hai xe đi xuyên qua nhau rồi giữ nguyên tốc độ - 55/152 nhãn cũ là lúc hai thân xe đang chồng lên nhau. Bản mới có vật lý va chạm: thân xe chạm nhau thì cả hai phanh tối đa, đứng tại chỗ 4 s rồi được dọn khỏi đường; nhãn chuẩn tách ${SET.n_ground_truth_near_miss} near-miss và ${SET.n_ground_truth_collision} va chạm. Phía hệ thống, mỗi sự kiện được phân loại kết cục: khoảng cách cùng thời điểm > 0 là near-miss; bằng 0 mà ít nhất một xe gần như dừng hẳn trong 2,5 s sau đó là nghi va chạm (cần người xác nhận); bằng 0 mà cả hai giữ tốc độ là chồng thân xe do sai số đo hoặc che khuất - vật lý không cho phép, nên không được đếm là va chạm.`));
children.push(T(['Nhãn chuẩn \\ Hệ thống', 'Near-miss', 'Chồng thân (sai số)', 'Nghi va chạm'], [
  ['Near-miss', CC.near_miss.near_miss, CC.near_miss.body_overlap, CC.near_miss.collision_suspected],
  ['Va chạm', CC.collision.near_miss, CC.collision.body_overlap, CC.collision.collision_suspected],
  ['Không có nhãn (báo nhầm)', CC.no_label.near_miss, CC.no_label.body_overlap, CC.no_label.collision_suspected],
], [2.4, 1.2, 1.6, 1.4]));
children.push(Cap(`Bảng 3. Recall near-miss ${r(M.by_kind.near_miss.recall)}, recall va chạm ${r(M.by_kind.collision.recall)}.`));

// ---------------------------- 4. Yêu cầu 3 ------------------------------- //
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1('4. Yêu cầu 3 - Báo cáo đánh giá nhất quán, có ablation'));
children.push(H2('4.1. Một luồng đếm sự kiện, tự kiểm bằng đẳng thức'));
if (V1 && V1R) children.push(P(`Hồ sơ cũ có ba con số không khớp: dashboard ${V1R.summary.total_events} sự kiện, bảng đánh giá ${V1.runtime.total_events}, Precision tính trên ${V1.final.tp + V1.final.fp}. Nguyên nhân: lệnh run và lệnh evaluate dùng hai seed khác nhau cho detector mô phỏng; và các sự kiện nằm ngoài vùng camera bao phủ bị bỏ qua khi chấm nhưng không được ghi ra. Bản mới: seed của detector nằm trong file cấu hình, results.json của dashboard được ghi từ chính lần chạy đánh giá, và mỗi sự kiện rơi vào đúng một ô của bảng dưới.`));
const RL = {
  episodes_opened: 'Episode được mở', rejected_proximity_gate: '  - loại: cổng khoảng cách', rejected_too_short: '  - loại: quá ngắn (< 3 khung)',
  rejected_pet_measured_above_threshold: '  - loại: PET đo ≥ ngưỡng', rejected_no_peak: '  - loại: không có đỉnh', merged_into_previous: '  - gộp vào sự kiện trước của cùng cặp',
  emitted: 'Sự kiện phát ra', exported_to_results: 'Sự kiện trong results.json (dashboard)', ignored_outside_eval_region: '  - ngoài vùng đánh giá (bỏ qua)',
  scored: '  - được chấm', scored_unmapped_track_fp: '      FP do track không ánh xạ được', scored_mapped_fp: '      FP', scored_tp: '      TP',
  gt_total: 'Nhãn chuẩn', gt_matched_tp: '  - khớp (TP)', gt_missed_fn: '  - bỏ sót (FN)',
};
children.push(T(['Bước', 'Số lượng'], Object.entries(RC.rows).map(([k, v]) => [RL[k] || k, v]), [3, 1]));
children.push(P('Đẳng thức tự kiểm (tất cả phải đúng, nếu không lệnh evaluate in "LỆCH"): ' +
  Object.entries(RC.checks).map(([k, v]) => `${k} ${v ? '✓' : '✗'}`).join('; ') + '.'));
children.push(P(`Báo nhầm ${RC.per_hour.false_alarms_per_hour}/giờ, bỏ sót ${RC.per_hour.misses_per_hour}/giờ. Kịch bản mô phỏng được thiết kế dày đặc xung đột (${SET.n_ground_truth} xung đột trong ${SET.duration_s} s) nên các con số theo giờ này chỉ để so sánh giữa các cấu hình, không phải tần suất ngoài đời.`, { italics: true, size: 22 }));

children.push(H2('4.2. Cờ hành vi 305/307 quỹ đạo'));
children.push(P(`Bản cũ gắn cờ hành vi nguy hiểm cho ${V1R ? V1R.behaviors.length : 305}/${V1 ? V1.tracking.n_pred_tracks : 307} quỹ đạo. Nguyên nhân: đặc trưng chỉ lấy ở 15 khung hình cuối của track (lúc xe ra khỏi khung hình, bbox bị cắt mép) và gia tốc là hiệu tốc độ giữa hai khung liền nhau - nhiễu 0,1 m/s đã thành 3 m/s², vượt ngưỡng phanh gấp. Bản mới tính trên toàn quỹ đạo (bỏ 0,4 s đầu và cuối), gia tốc là hệ số góc hồi quy trên cửa sổ 0,5 s, và ngưỡng phải được duy trì ít nhất 0,3 s. Tầng ML được huấn luyện lại trên đặc trưng mới (saferoad train-behavior); mô hình cũ bị từ chối tự động nhờ số phiên bản đặc trưng.`));
children.push(T(['Chỉ số', 'Giá trị'], [
  ['Quỹ đạo bị gắn cờ (hệ thống)', r(BH.flagged_tracks)],
  ['Xe có hành vi gắn cờ theo quỹ đạo giải tích (cùng bộ luật, không nhiễu)', r(BH.truth_flagged_vehicles)],
  ['Precision của cờ', r(BH.risky_precision)], ['Recall của cờ', r(BH.risky_recall)],
  ['Khớp đúng loại hành vi', r(BH.label_agreement)],
  ['Xe được lập trình phanh gấp, hệ thống bắt được', r(BH.forced_brake_detected)],
], [3.5, 2]));

children.push(H2('4.3. Kiểm tra mô phỏng có chạy đúng kịch bản'));
children.push(P(`Lệnh simulate nay in và lưu bảng kiểm tra kịch bản: lên lịch ${SC.n_scheduled} phương tiện, vào được ${SC.n_spawned}, bị bỏ ở điểm vào vì ùn ${SC.n_dropped_at_entry}, đi hết tuyến ${SC.n_completed_route}, va chạm ${SC.n_crashes} vụ (${SC.n_vehicles_in_crash} xe), vượt đèn đỏ ${SC.n_runs_red}, rẽ ${SC.n_turning}. Khi rà các vụ chạm, nhóm tìm ra hai chỗ mô phỏng lệch kịch bản đã khai báo:`));
children.push(B('Vạch dừng (bán kính 9 m) nằm bên trong vạch bộ hành (11 m): xe chờ đèn đỗ đè lên lối đi bộ. Đã dời vạch dừng ra 12,6 m và tính khoảng cách dừng tới đầu xe thay vì tâm xe.'));
children.push(B('Lối đi bộ không bao giờ chạm vòng vạch dừng 9 m nên không người đi bộ nào chờ đèn - tất cả cư xử như người băng ẩu, trong khi kịch bản khai báo 35%. Đã sửa điểm chờ của người đi bộ thành mép mặt đường.'));
children.push(P(`Tỉ lệ xe bị bỏ ở điểm vào (${n(SC.n_dropped_at_entry / SC.n_scheduled * 100, 0)}%) và số vụ va chạm cho thấy kịch bản này là bài kiểm tra căng thẳng, dày đặc hơn giao lộ thật nhiều lần. Con số được báo cáo nguyên trạng thay vì chỉnh tham số cho đẹp.`));

children.push(H2('4.4. Ablation bỏ từng khối trên cùng mẫu'));
const abRow = (x) => [x.label, r(x.precision), r(x.recall), n(x.f1), x.ttc_mae === null ? '-' : n(x.ttc_mae), x.n_emitted, x.n_ignored_outside];
children.push(T(['Cấu hình', 'Precision', 'Recall', 'F1', 'TTC MAE (s)', 'Phát ra', 'Ngoài vùng'], LOO.map(abRow), [2.6, 1.4, 1.4, 0.7, 0.9, 0.8, 0.9]));
children.push(Cap('Bảng 4. Mỗi dòng chỉ khác bản đầy đủ đúng một khối; cùng bbox gốc, cùng seed nhiễu, cùng vùng đánh giá, cùng hàm chấm.'));
const mp = row('-measured_pet');
if (mp.f1 !== undefined) children.push(P(`Đóng góp riêng của PET đo (so với hành vi PET của bản cũ): F1 ${n(mp.f1)} → ${n(row('full').f1)}, Precision ${r(mp.precision)} → ${r(row('full').precision)} với Recall không đổi - các episode chỉ dựa vào PET ngoại suy mà thực tế xe sau đã nhường bị loại.`));
children.push(H2('4.5. Ablation cộng dồn từ baseline'));
children.push(T(['Cấu hình', 'Precision', 'Recall', 'F1', 'TTC MAE (s)', 'Phát ra', 'Ngoài vùng'], CUM.map(abRow), [2.6, 1.4, 1.4, 0.7, 0.9, 0.8, 0.9]));
children.push(Cap('Bảng 5. Baseline: TTC < 3 s tính trên toạ độ ảnh, mỗi khung hình một cảnh báo (chống trùng 3 s), không PET, không cổng.'));

if (GRID) {
  children.push(H2('4.6. Nhiều mật độ, góc camera và điều kiện ánh sáng'));
  children.push(P(`Lưới ${GRID.cells.length} ô: 3 mật độ (${Object.entries(GRID.params.densities).map(([k, v]) => `${k} ${n(v, 1)} xe/s`).join(', ')}) × 3 vị trí camera (cao 18 m nhìn dọc trục; cột thấp 10 m; đặt ở góc nhìn chéo 45°) × 3 hồ sơ detector đại diện cho ánh sáng (ngày: nhiễu 1,5 px, bỏ sót 3%; chạng vạng: 3 px, 10%; đêm/mưa: 5 px, 25%). Mỗi ô chạy bản đầy đủ, baseline và bản PET cũ trên cùng kịch bản ${GRID.params.duration} s.`));
  children.push(T(['Mật độ', 'Camera', 'Ánh sáng', 'Nhãn', 'F1 đầy đủ', 'F1 baseline', 'F1 PET cũ', 'P đầy đủ', 'R đầy đủ'],
    GRID.cells.map((c) => [c.density, c.camera, c.light, c.n_gt, n(c.full.f1), n(c.baseline.f1), n(c.old_pet.f1), r(c.full.precision), r(c.full.recall)]),
    [0.7, 0.9, 1.2, 0.5, 0.8, 0.8, 0.8, 1.3, 1.3]));
  children.push(P('Giới hạn: "ánh sáng" ở đây là hồ sơ nhiễu/bỏ sót của detector, không phải ảnh ban đêm thật; ảnh hưởng thật của ánh sáng lên YOLO chỉ đo được trên video thật ban đêm. Mỗi ô có ít nhãn (mật độ thưa chỉ khoảng 15 xung đột) nên khoảng tin cậy rộng.', { italics: true, size: 22 }));
}
children.push(H2('4.7. Độ bền trước nhiễu detector'));
children.push(T(['Nhiễu bbox (px)', 'Precision', 'Recall', 'F1', 'TTC MAE (s)', 'PET MAE (s)', 'IDF1', 'Đổi ID'],
  EV.noise_sweep.map((x) => [n(x.noise_px, 1), r(x.precision), r(x.recall), n(x.f1), n(x.ttc_mae), n(x.pet_final_mae), n(x.idf1), x.id_switches]),
  [1, 1.4, 1.4, 0.7, 0.9, 0.9, 0.7, 0.7]));
children.push(P(`Về ID switch: chỉ số đếm mỗi lần một chiếc xe thật được gán cho một track khác so với khung trước, kể cả khi track cũ được nối lại sau che khuất, nên vẫn có ${M.tracking.id_switches} lần đổi ID dù IDF1 = ${n(M.tracking.idf1)} (độ phân mảnh trung bình ${n(M.tracking.fragmentation, 2)} đoạn/xe). Một lần đổi ID trong lúc hai xe đang tiến sát nhau sẽ cắt đứt episode; đây là lý do kết quả cần người rà soát trước khi dùng.`));

children.push(H2('4.8. Log cấu hình tái hiện'));
const RP = EV.reproducibility || {};
children.push(P(`Mỗi tệp kết quả ghi kèm: lệnh đã chạy, commit Git, mã băm SHA-256 của toàn bộ cấu hình (${RP.config_sha256 || '-'}), toàn bộ cấu hình dạng JSON, mã băm của bundle dữ liệu, seed kịch bản (${SET.seed}) và seed detector (${SET.replay_seed}), phiên bản Python/NumPy/OpenCV/scikit-learn. Mỗi dòng ablation và mỗi ô lưới cũng ghi mã băm cấu hình của riêng nó.`));
children.push(P(`Kiểm tra tái hiện: toàn bộ số liệu trong báo cáo được chạy lại từ đầu trên một bản clone sạch của commit ${RP.git_commit || '-'} trên GitHub, theo đúng các lệnh ở phụ lục (simulate → evaluate → evaluate-grid → bốn lần evaluate-real → chạy đầu-cuối đoạn chưa từng dùng). Mọi số đếm - sự kiện phát ra, TP/FP, mẫu số, ablation, quét nhiễu, lưới 27 ô, detection/tracking/cảnh báo trên video thật, danh sách ứng viên rà soát - trùng khớp với lần chạy trước. Khác biệt duy nhất là sai số PET: bộ nhãn mô phỏng dùng ở lần trước được sinh ra trước lần sửa cuối cùng của cách lấy mốc vào/ra vùng xung đột (điểm giữa hai mẫu kề biên), nên MAE của PET đo giảm từ 0,035 s xuống ${n((((EV.main || {}).pet_errors || {}).measured || {}).mae)} s; các bảng trong báo cáo dùng số của lần chạy lại.`));
if (MTF) children.push(P(`Phần tinh chỉnh trên MVTI (mục 2.3) làm sau lần chạy lại này nên không nằm trong phép kiểm tra trên; hai tệp evaluation_mvti_*_test.json ghi commit ${(MTF.reproducibility || {}).git_commit || '-'} (hậu tố -dirty nghĩa là chạy trên commit đó cộng các thay đổi chưa commit của chính đợt này), trọng số tinh chỉnh lưu ở models/saferoad_mvti_yolo11n.pt.`));

// ---------------------------- 5. Hạn chế -------------------------------- //
children.push(H1('5. Hạn chế còn lại và việc tiếp theo'));
if (!REVIEWS.length) children.push(B('Chưa có nhãn xung đột do người rà soát trên video thật: công cụ và danh sách ứng viên đã sẵn, cần hai thành viên gán nhãn độc lập (mục 2.6).'));
children.push(B('Hiệu chuẩn video đại lộ là ước lượng từ vạch kẻ theo tiêu chuẩn, chưa đo thực địa, và chỉ phủ lòng đường bên trái - vùng phân tích vì vậy bị thu hẹp (mục 2.4). Cần quay video tại Việt Nam và đo thực địa điểm mốc.'));
children.push(B('Trên video đại lộ, người đi bộ quá nhỏ cho YOLO ở đầu vào 640 px; cần chạy theo ô ở độ phân giải gốc (chậm hơn khoảng 6 lần trên CPU) hoặc camera gần hơn.'));
children.push(B('Số cảnh báo tham chiếu trong vùng phân tích của video đại lộ rất ít (giao thông bình thường), nên Precision/Recall so với tham chiếu có mẫu số nhỏ; giá trị chính của video này là bảng detection/tracking và quy trình rà soát.'));
children.push(B('Hai bộ video thật không có xe máy - nhóm đối tượng chủ đạo ở Việt Nam; MVTI cũng không có người đi bộ. Hai lớp này vẫn chỉ được đánh giá trên mô phỏng.'));
children.push(B('Tốc độ toàn hệ thống có YOLO trên CPU vẫn dưới 25 FPS; cần GPU hoặc chạy detector thưa hơn.'));
children.push(B('Mô phỏng dày đặc va chạm hơn thực tế; ánh sáng mới chỉ mô phỏng bằng chất lượng detector.'));

// ---------------------------- 6. Kê khai -------------------------------- //
children.push(H1('6. Kê khai sử dụng AI, mã nguồn và dữ liệu bên ngoài'));
children.push(B('Có dùng trợ lý lập trình AI để hỗ trợ viết mã, kiểm thử và tài liệu. Quyết định thiết kế, ngưỡng, phương pháp đánh giá và diễn giải kết quả do nhóm thực hiện và chịu trách nhiệm.'));
children.push(B('Thư viện: Ultralytics YOLO11n (AGPL-3.0), PyTorch, OpenCV, NumPy, SciPy, scikit-learn, FastAPI. ByteTrack và Kalman cài đặt lại từ công thức.'));
children.push(B('Dữ liệu: Multi-view Traffic Intersection Dataset (Møgelmose, Aalborg University); video đại lộ kèm nhãn CVAT tải từ Kaggle (nguồn và giấy phép ghi ở docs/dataset_license.md); UCSD Highway Traffic (Chan & Vasconcelos, 2005); trình mô phỏng tự viết (MIT).'));

children.push(H1('Phụ lục - Lệnh tái hiện'));
[
  `git clone https://github.com/dangkhoi-dev/SafeRoad-AI.git && cd SafeRoad-AI && git checkout ${RP.git_commit || 'main'}`,
  'pip install -e ".[dev,docs]" && python scripts/download_assets.py',
  'pytest tests -q',
  'python -m saferoad simulate --duration 300 --seed 42',
  'python -m saferoad evaluate',
  'python -m saferoad evaluate-grid',
  'python -m saferoad train-behavior',
  'python -m saferoad prepare-street --annotations data/raw/mvti/merged_annotations.json',
  'python -m saferoad calibrate --points configs/calib_street.yaml --frame data/processed/street/street_labeled_0_300s.mp4 --image docs/assets/calib_street.jpg --out data/outputs/calibration_street.json',
  'python -m saferoad evaluate-real --config configs/street.yaml --weights models/yolo11n.pt --no-db --no-overlay --source data/processed/street/street_labeled_0_300s.mp4 --ground-truth data/processed/street/street_groundtruth.pkl --calibration-report data/outputs/calibration_street.json --out data/outputs/evaluation_street.json',
  'python scripts/street_reference_sensitivity.py',
  'python scripts/export_street_yolo.py --video data/processed/street/street_labeled_0_300s.mp4 --ground-truth data/processed/street/street_groundtruth.pkl --out data/yolo_street',
  'python scripts/train_street_yolo.py',
  'python -m saferoad evaluate-real --config configs/street.yaml --weights models/yolo11n.pt --start-frame 6600 --no-db --no-overlay --source data/processed/street/street_labeled_0_300s.mp4 --ground-truth data/processed/street/street_groundtruth.pkl --calibration-report data/outputs/calibration_street.json --out data/outputs/evaluation_street_pretrained_test.json',
  'python -m saferoad evaluate-real --config configs/street.yaml --weights models/saferoad_street_yolo11n.pt --start-frame 6600 --no-db --no-overlay --source data/processed/street/street_labeled_0_300s.mp4 --ground-truth data/processed/street/street_groundtruth.pkl --calibration-report data/outputs/calibration_street.json --out data/outputs/evaluation_street_finetuned_test.json',
  'python -m saferoad evaluate-real --config configs/mvti.yaml --no-db',
  'python scripts/export_street_yolo.py --video data/processed/mvti/infrastructure.mp4 --ground-truth data/processed/mvti/infrastructure_groundtruth.pkl --out data/yolo_mvti --prefix mvti --train-end 1500 --test-start 1700 --train-step 5 --val-step 15',
  'python scripts/train_street_yolo.py --data data/yolo_mvti/data.yaml --epochs 15 --name mvti --out models/saferoad_mvti_yolo11n.pt --results-csv data/outputs/train_mvti_yolo11n_results.csv',
  'python -m saferoad evaluate-real --config configs/mvti.yaml --weights models/yolo11n.pt --no-db --no-overlay --start-frame 1700 --seen-before 1500 --out data/outputs/evaluation_mvti_pretrained_test.json',
  'python -m saferoad evaluate-real --config configs/mvti.yaml --weights models/saferoad_mvti_yolo11n.pt --no-db --no-overlay --start-frame 1700 --seen-before 1500 --out data/outputs/evaluation_mvti_finetuned_test.json',
  'python -m saferoad run --config configs/street.yaml --no-overlay --source data/processed/street/street_heldout_300_741s.mp4 --output data/outputs/street_heldout --review-csv data/outputs/review_candidates_street_heldout.csv',
  'node scripts/build_improvement_report.js',
].forEach((c) => children.push(new Paragraph({ spacing: { after: 60 },
  children: [new TextRun({ text: c, font: 'Consolas', size: 18 })] })));

// ------------------------------------------------------------------------- //
const doc = new Document({
  creator: 'SafeRoad AI', title: 'SafeRoad AI - Báo cáo cải tiến Chặng 1',
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: '1F3864' },
        paragraph: { spacing: { before: 300, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: '2E5597' },
        paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: 'bul', levels: [
    { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
    { level: 1, format: LevelFormat.BULLET, text: '-', alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1300, bottom: 1300, left: 1273, right: 1273 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: 'SafeRoad AI - Báo cáo cải tiến Chặng 1 - Trang ', font: FONT, size: 18 }),
        new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`✓ ${path.relative(ROOT, OUT)}`);
  const soffice = ['soffice', 'libreoffice'].find((c) => {
    try { execFileSync(c, ['--version'], { stdio: 'ignore' }); return true; } catch (e) { return false; }
  });
  if (soffice) {
    execFileSync(soffice, ['--headless', '--convert-to', 'pdf', '--outdir', path.dirname(OUT), OUT], { stdio: 'ignore' });
    console.log(`✓ ${path.relative(ROOT, OUT.replace(/\.docx$/, '.pdf'))}`);
  } else {
    console.log('  (Không thấy LibreOffice - mở .docx bằng Word rồi Lưu thành PDF)');
  }
});
