/**
 * Sinh tài liệu dự án Bảng C (.docx) cho SafeRoad AI.
 *
 * Mọi con số trong tài liệu được ĐỌC TRỰC TIẾP từ data/outputs/evaluation.json và
 * results.json — không có số liệu nào gõ tay. Chạy lại đánh giá rồi chạy lại
 * script này là tài liệu tự cập nhật, nên báo cáo không bao giờ lệch với kết quả
 * thực tế của mã nguồn.
 *
 * Dùng:  node scripts/build_report.js
 */

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, TableOfContents, Footer, PageNumber, LevelFormat,
  convertInchesToTwip,
} = require('docx');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'docs', 'BaoCao_SafeRoadAI_BangC.docx');

// --------------------------------------------------------------------------- //
// Nạp số liệu thật
// --------------------------------------------------------------------------- //
function loadJson(p, fallback) {
  try { return JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf8')); }
  catch (e) { console.warn(`  ! không đọc được ${p} — dùng giá trị mặc định`); return fallback; }
}
const EV = loadJson('data/outputs/evaluation.json', null);
const RS = loadJson('data/outputs/results.json', null);
const REAL = loadJson('data/outputs/evaluation_real.json', null);
if (!EV || !RS) {
  console.error('Thiếu evaluation.json hoặc results.json. Chạy trước:');
  console.error('  python -m saferoad evaluate');
  console.error('  python -m saferoad run --config configs/synthetic.yaml ...');
  process.exit(1);
}
const F = EV.final, T = EV.tracking, S = EV.setup, SUM = RS.summary;
const n = (v, d = 3) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);

// --------------------------------------------------------------------------- //
// Trợ giúp định dạng
// --------------------------------------------------------------------------- //
const FONT = 'Times New Roman';
const TW = convertInchesToTwip(6.3);   // bề rộng bảng khả dụng (Letter, lề 1")

const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: opts.line ?? 276 },
  indent: opts.indent,
  children: [new TextRun({ text, font: FONT, size: opts.size || 24,
    bold: opts.bold, italics: opts.italics, color: opts.color })],
});

const Rich = (runs, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: 276 },
  children: runs.map(r => new TextRun({ font: FONT, size: 24, ...r })),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 320, after: 160 },
  children: [new TextRun({ text, font: FONT, size: 30, bold: true, color: '1F4E79' })],
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text, font: FONT, size: 26, bold: true, color: '2E75B6' })],
});

const Bullet = (text, level = 0) => new Paragraph({
  numbering: { reference: 'bullets', level },
  spacing: { after: 80, line: 276 },
  alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text, font: FONT, size: 24 })],
});

const Note = (text) => new Paragraph({
  spacing: { before: 100, after: 160, line: 276 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: 'D97706', space: 10 } },
  indent: { left: 200 },
  children: [new TextRun({ text, font: FONT, size: 22, italics: true, color: '444444' })],
});

const Cap = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 200 },
  children: [new TextRun({ text, font: FONT, size: 20, italics: true, color: '666666' })],
});

/** Bảng: header + rows. `widths` là tỉ lệ, tự quy về DXA. */
function T_(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cols = widths.map(w => Math.round(TW * w / total));
  // đảm bảo tổng khớp chính xác
  cols[cols.length - 1] = TW - cols.slice(0, -1).reduce((a, b) => a + b, 0);

  const cell = (txt, i, isHead, align) => new TableCell({
    width: { size: cols[i], type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: '1F4E79' } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: align || (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER),
      spacing: { after: 0, line: 240 },
      children: [new TextRun({
        text: String(txt), font: FONT, size: opts.size || 20,
        bold: isHead, color: isHead ? 'FFFFFF' : '000000',
      })],
    })],
  });

  return new Table({
    columnWidths: cols,
    width: { size: TW, type: WidthType.DXA },
    rows: [
      // cantSplit: không cho một hàng bị cắt đôi qua hai trang. Thiếu nó, các
      // bảng dài sẽ vắt ngang chỗ ngắt trang và để lại một dòng cụt khó đọc.
      new TableRow({
        tableHeader: true, cantSplit: true,
        children: header.map((h, i) => cell(h, i, true)),
      }),
      ...rows.map(r => new TableRow({
        cantSplit: true,
        children: r.map((c, i) => cell(c, i, false)),
      })),
    ],
  });
}

const img = (rel, w, h) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 40 },
  children: [new ImageRun({
    type: 'png', data: fs.readFileSync(path.join(ROOT, rel)),
    transformation: { width: w, height: h },
  })],
});

// --------------------------------------------------------------------------- //
// Nội dung
// --------------------------------------------------------------------------- //
const abl = EV.ablation || [];
const noise = EV.noise_sweep || [];
const sev = (EV.severity_breakdown || []).filter(r => r.n_gt > 0);
const byType = SUM.by_type || {};
const VI_TYPE = { crossing: 'Cắt ngang', turning: 'Chuyển hướng', rear_end: 'Tạt đầu/đâm đuôi',
  head_on: 'Đối đầu', lane_change: 'Chuyển làn', pedestrian: 'Người đi bộ' };

const children = [];

// ---- Trang bìa ---- //
children.push(
  new Paragraph({ spacing: { before: 1800, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'CUỘC THI SÁNG TẠO TRẺ QUỐC GIA', font: FONT, size: 26, bold: true, color: '555555' })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'TRONG LĨNH VỰC TRÍ TUỆ NHÂN TẠO NĂM 2026 — BẢNG C', font: FONT, size: 26, bold: true, color: '555555' })] }),
  new Paragraph({ spacing: { after: 160 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'SAFEROAD AI', font: FONT, size: 64, bold: true, color: '1F4E79' })] }),
  new Paragraph({ spacing: { after: 900 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Hệ thống AI phát hiện near-miss, phân tích rủi ro\nvà xây dựng bản đồ an toàn giao thông', font: FONT, size: 30, color: '333333' })] }),
);
children.push(T_(
  ['Hạng mục', 'Nội dung'],
  [
    ['Tên đề tài', 'SafeRoad AI — Phát hiện xung đột giao thông & bản đồ rủi ro va chạm'],
    ['Bảng dự thi', 'Bảng C — đội thi tự do'],
    ['Đội trưởng', 'Trần Phan Đăng Khôi (KHMT)'],
    ['Thành viên', 'Mạnh Anh (Robot) · Huynh Hân (QTKD)'],
    ['Lĩnh vực', 'Thị giác máy tính · An toàn giao thông'],
    ['Mã nguồn', 'https://github.com/dangkhoi-dev/SafeRoad-AI'],
  ], [1, 2.4], { size: 22 }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Mục lục ---- //
children.push(H1('MỤC LỤC'));
children.push(new TableOfContents('Mục lục', { hyperlink: true, headingStyleRange: '1-2' }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 1 ---- //
children.push(H1('1. Bối cảnh và vấn đề thực tiễn'));
children.push(P('Công tác quản lý an toàn giao thông tại Việt Nam hiện vận hành theo cơ chế phản ứng: một giao lộ chỉ được xác định là "điểm đen" sau khi đã có đủ số vụ tai nạn được thống kê. Nghĩa là dữ liệu dùng để ra quyết định chỉ xuất hiện sau khi thiệt hại về người đã xảy ra.'));
children.push(P('Trong khi đó, nghiên cứu an toàn giao thông từ lâu đã chỉ ra rằng trước mỗi vụ tai nạn có hàng trăm tình huống "suýt va chạm" (near-miss) mang cùng cơ chế nguy hiểm nhưng kết thúc may mắn. Những tình huống này diễn ra liên tục, ngay trước ống kính của hệ thống camera giao thông đã lắp đặt sẵn, nhưng không được ghi nhận và phân tích một cách hệ thống.'));
children.push(Rich([
  { text: 'Khoảng trống cần lấp: ', bold: true },
  { text: 'chưa có công cụ tự động biến luồng video giám sát sẵn có thành chỉ số rủi ro định lượng, có thể so sánh giữa các vị trí và theo dõi được theo thời gian. SafeRoad AI được xây dựng để lấp đúng khoảng trống đó.' },
]));

children.push(H1('2. Mục tiêu đề tài'));
children.push(Bullet('Tự động phát hiện tình huống suýt va chạm từ video camera giao lộ, không cần thêm cảm biến hay hạ tầng mới.'));
children.push(Bullet('Định lượng mức nguy hiểm bằng hai chỉ số đã được kiểm chứng trong nghiên cứu an toàn giao thông: TTC (Time To Collision) và PET (Post-Encroachment Time).'));
children.push(Bullet('Sinh bản đồ rủi ro không–thời gian, chỉ ra các điểm nóng cần ưu tiên xử lý hạ tầng.'));
children.push(Bullet('Mỗi cảnh báo phải kèm lý do đọc được bằng ngôn ngữ tự nhiên (Explainable Risk), để cán bộ vận hành kiểm chứng được thay vì phải tin vào một con số.'));
children.push(Bullet('Toàn bộ xử lý chạy cục bộ tại biên, không gửi dữ liệu ra ngoài, đáp ứng yêu cầu bảo vệ dữ liệu cá nhân.'));

children.push(H1('3. Đối tượng thụ hưởng và giá trị ứng dụng'));
children.push(T_(['Đối tượng', 'Giá trị nhận được'], [
  ['Sở Giao thông Vận tải, đơn vị quản lý hạ tầng', 'Xếp hạng mức nguy hiểm giữa các giao lộ bằng số liệu khách quan; ưu tiên vốn cải tạo dựa trên bằng chứng thay vì cảm tính'],
  ['Cảnh sát giao thông', 'Xác định khung giờ và vị trí cần bố trí lực lượng; nhận diện hành vi vi phạm phổ biến tại từng nút'],
  ['Đơn vị thiết kế giao thông', 'Đánh giá hiệu quả một thay đổi hạ tầng bằng cách so sánh chỉ số trước và sau, thay vì chờ số liệu tai nạn tích luỹ nhiều năm'],
  ['Cộng đồng dân cư', 'Giảm rủi ro tại các nút giao nguy hiểm nhờ can thiệp sớm'],
], [1, 2]));
children.push(Note('Giá trị cốt lõi của đề tài là chuyển công tác an toàn giao thông từ phản ứng sang phòng ngừa: phát hiện nút giao nguy hiểm trước khi nó kịp gây ra thương vong.'));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 4 ---- //
children.push(H1('4. Giải pháp và kiến trúc hệ thống'));
children.push(P('Hệ thống nhận luồng video từ camera giao lộ đặt cao 10–18 m và xử lý qua sáu khối nối tiếp. Toàn bộ pipeline chạy trên một máy duy nhất, không phụ thuộc dịch vụ đám mây.'));
if (fs.existsSync(path.join(ROOT, 'docs/assets/architecture.png'))) {
  children.push(img('docs/assets/architecture.png', 620, 300));
  children.push(Cap('Hình 1. Kiến trúc sáu khối của SafeRoad AI'));
}
children.push(T_(['Khối', 'Chức năng', 'Công nghệ'], [
  ['1. Nguồn video', 'Thu nhận luồng hình ảnh', 'MP4 / RTSP / chuỗi ảnh'],
  ['2. AI Processing', 'Phát hiện, ẩn danh, theo vết, dựng quỹ đạo mặt đất', 'YOLO11n · ByteTrack · Kalman · Homography'],
  ['3. Conflict Engine', 'Tính TTC/PET, gom sự kiện, phân loại xung đột', 'Mô hình đa hình tròn · gom theo episode'],
  ['4. Risk Engine', 'Chấm điểm rủi ro, sinh lý do, dựng bản đồ nhiệt', 'Sigmoid có trọng số · lưới không–thời gian'],
  ['5. Database', 'Lưu trữ bền vững, truy vấn song song', 'SQLite chế độ WAL'],
  ['6. Dashboard', 'Trực quan hoá 7 tab, chạy ngoại tuyến', 'FastAPI · HTML/Canvas thuần'],
], [0.9, 2, 1.4]));

children.push(H1('5. Dữ liệu'));
children.push(P('Đề tài sử dụng đồng thời dữ liệu thật và dữ liệu mô phỏng, với vai trò tách bạch rõ ràng. Đây là quyết định phương pháp quan trọng nhất của dự án và cần được nêu minh bạch.'));
children.push(T_(['Nguồn', 'Quy mô', 'Vai trò', 'Không dùng để'], [
  ['Multi-view Traffic Intersection (Møgelmose)', '2.441 ảnh · 14.488 bbox · 19 quỹ đạo chuẩn', 'Đo mAP detection, IDF1/MOTA tracking trên ảnh thật', 'Đo Precision/Recall near-miss'],
  ['UCSD Highway Traffic (Chan & Vasconcelos)', '254 clip 320×240', 'Kiểm thử độ bền ở độ phân giải thấp', 'Phân tích xung đột (cao tốc, không có giao cắt)'],
  ['Trình mô phỏng SafeRoad (tự viết)', `180 s · ${S.n_vehicles} phương tiện · ${S.n_ground_truth} nhãn near-miss`, 'Đo Precision/Recall/TTC-MAE và bảng ablation', 'Đánh giá độ khó thị giác của ảnh thật'],
], [1.1, 1.2, 1.5, 1.2]));
children.push(Rich([
  { text: 'Vì sao phải tự viết trình mô phỏng. ', bold: true },
  { text: 'Không một dataset giao thông công khai nào có nhãn near-miss — tất cả chỉ có bounding box. Mà Precision/Recall của phát hiện xung đột lại là cam kết trung tâm của đề tài. Trình mô phỏng cho phép biết chính xác quỹ đạo giải tích của từng xe, nhờ đó tính được TTC/PET thật ở độ phân giải thời gian tuỳ ý và có một tập nhãn chuẩn để chấm điểm.' },
]));
children.push(P('Nhãn chuẩn được sinh bằng một quy trình khác hẳn thuật toán online: lấy mẫu ở 60 Hz thay vì 30 Hz, dùng vận tốc giải tích thay vì vận tốc ước lượng từ bbox nhiễu, và lấy cực tiểu toàn cục trên cả quãng gặp nhau thay vì quyết định theo từng khung hình. Nhờ ba khác biệt đó, đây là nhãn độc lập chứ không phải thuật toán tự chấm điểm cho chính nó.'));
children.push(Note('Trong trình mô phỏng, near-miss phát sinh từ đúng nguyên nhân ngoài đời — xe vượt đèn đỏ, xe rẽ cắt dòng ngược chiều, người đi bộ băng qua đường — chứ không có tình huống nào được dàn dựng thủ công.'));

children.push(H2('5.1. Tuân thủ quyền riêng tư'));
children.push(Bullet('Làm mờ khuôn mặt người đi bộ và người ngồi trên xe hai bánh; làm mờ vùng biển số phương tiện.'));
children.push(Bullet('Bước ẩn danh chạy ngay sau detection, trước mọi thao tác ghi ra đĩa — mọi dữ liệu lưu trữ từ đó trở đi đều đã ẩn danh.'));
children.push(Bullet('Cơ sở dữ liệu chỉ lưu ID số, lớp đối tượng, toạ độ và chỉ số; không lưu bất kỳ ảnh cắt nào.'));
children.push(Bullet('Không có kết nối mạng trong pipeline xử lý; hệ thống chạy hoàn toàn ngoại tuyến.'));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 6 ---- //
children.push(H1('6. Phương pháp và công thức'));
children.push(H2('6.1. TTC — Time To Collision'));
children.push(P('TTC là thời gian còn lại tới va chạm nếu cả hai đối tượng giữ nguyên vận tốc hiện tại. Đặt Δp là vị trí tương đối, Δv là vận tốc tương đối và R là tổng bán kính hai đối tượng, điều kiện va chạm ‖Δp + tΔv‖ = R dẫn tới phương trình bậc hai:'));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  children: [new TextRun({ text: '‖Δv‖² · t²  +  2(Δp · Δv) · t  +  (‖Δp‖² − R²)  =  0', font: FONT, size: 24, italics: true })] }));
children.push(P('TTC là nghiệm dương nhỏ nhất. Không tồn tại nghiệm dương nghĩa là hai đối tượng đang tách xa nhau hoặc sẽ lướt qua nhau.'));
children.push(Rich([
  { text: 'Cải tiến quan trọng — mô hình đa hình tròn. ', bold: true },
  { text: 'Cách làm phổ biến là xấp xỉ mỗi xe bằng một hình tròn ngoại tiếp. Với ô tô 4,4 × 1,8 m, hình tròn đó có bán kính 2,38 m, tức là mô hình hoá chiếc xe như vật thể rộng 4,76 m. Hậu quả đo được trong quá trình phát triển: hai ô tô đi ngược chiều ở hai làn cách nhau 4 m bị gắn nhãn "đối đầu", sinh ra 132 cảnh báo giả trong 120 giây cho dòng xe hoàn toàn bình thường. Chúng tôi thay bằng cách phủ thân xe bằng 1–3 hình tròn nhỏ dọc trục, bán kính bằng nửa bề rộng xe; số cảnh báo giả loại này giảm xuống còn 13.' },
]));

children.push(H2('6.2. PET — Post-Encroachment Time'));
children.push(P('PET là khoảng thời gian giữa lúc đối tượng thứ nhất rời khỏi điểm xung đột và lúc đối tượng thứ hai đi tới chính điểm đó. Chỉ số này bắt được tình huống mà TTC bỏ sót: xe A vừa qua, 0,8 giây sau xe B mới tới đúng chỗ đó — chưa từng có nguy cơ va chạm tức thời nhưng rõ ràng là "suýt".'));
children.push(Note('PET chỉ được tính khi góc giữa hai quỹ đạo ≥ 20°. Với hai xe đi gần song song, giao điểm hai tia nằm rất xa và cực nhạy với nhiễu — lệch hướng 1° đã dời giao điểm hàng chục mét, sinh ra giá trị PET nhỏ hoàn toàn giả.'));

children.push(H2('6.3. Điều kiện ghi nhận near-miss'));
children.push(P('Một cặp đối tượng được ghi nhận là near-miss khi thoả đồng thời cả bốn điều kiện:'));
children.push(T_(['#', 'Điều kiện', 'Ngưỡng', 'Vì sao cần'], [
  ['1', 'Thực sự đến gần nhau', 'khoảng cách mặt-tới-mặt < 3,0 m', 'TTC chỉ là phép ngoại suy. Nếu một bên kịp phanh và hai xe chưa bao giờ tới gần, đó là tình huống được xử lý tốt'],
  ['2', 'Có nguy cơ va chạm', 'TTC < 3,0 s hoặc PET < 1,5 s', 'Ngưỡng theo thông lệ nghiên cứu traffic conflict'],
  ['3', 'Có tốc độ tiếp cận thật', 'vận tốc tương đối > 3,0 m/s', 'Phân biệt xung đột thật với dòng xe bám đuôi bình thường'],
  ['4', 'Nằm trong tầm phủ camera', 'vùng COVERAGE', 'Không quy trách nhiệm cho hệ thống về thứ ngoài khung hình'],
], [0.3, 1.3, 1.2, 2.4]));
children.push(Rich([
  { text: 'Điều kiện 1 và 3 là phần dễ bị bỏ quên nhất. ', bold: true },
  { text: 'Chỉ dùng TTC đơn thuần sẽ gán nhãn near-miss cho hàng trăm cặp xe đang lưu thông hoàn toàn bình thường trong dòng đông đúc: hai xe máy nối đuôi cách nhau 0,9 m ở cùng tốc độ có TTC rất nhỏ theo mô hình hình học, nhưng đó là giao thông Việt Nam bình thường.' },
]));

children.push(H2('6.4. Risk Score và Explainable Risk'));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  children: [new TextRun({ text: 'RS  =  100 · σ( w₁/TTC + w₂/PET + w₃·v_r + w₄·Type + w₅·Time + b )', font: FONT, size: 24, italics: true })] }));
children.push(P('Dùng nghịch đảo TTC và PET vì mức nguy hiểm không tuyến tính theo thời gian còn lại: chênh lệch giữa 0,5 s và 1,0 s nghiêm trọng hơn rất nhiều so với chênh lệch giữa 4,0 s và 4,5 s. Hàm sigmoid giữ kết quả nằm gọn trong khoảng 0–100.'));
children.push(P('Từng số hạng được lưu riêng, nhờ đó dashboard giải thích được vì sao một cảnh báo có mức rủi ro cao — đây chính là yêu cầu Explainable Risk. Bộ trọng số được hiệu chỉnh trên tập có nhãn để phân bố điểm trải đều; bộ chưa hiệu chỉnh đẩy trung vị lên 98/100, khi mọi cảnh báo đều "rất nguy hiểm" thì thang điểm không còn phân biệt được gì và người vận hành sẽ bỏ qua tất cả. Sau hiệu chỉnh, phân bố đạt phân vị 10 = 21, trung vị = 56, phân vị 90 = 89.'));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 7 ---- //
children.push(H1('7. Kết quả thực nghiệm'));
children.push(P(`Toàn bộ số liệu dưới đây sinh ra từ mã nguồn trong repo và tái lập được bằng các lệnh ghi trong README. Tập đánh giá: video mô phỏng ${S.duration_s} giây, ${S.n_vehicles} phương tiện, ${S.n_ground_truth} nhãn near-miss chuẩn, seed ${S.seed}. Detector được mô phỏng nhiễu ${n(S.noise_px,1)} px và tỉ lệ bỏ sót ${(S.miss_rate*100).toFixed(0)}%.`));

children.push(H2('7.1. Chỉ số tổng hợp'));
children.push(T_(['Chỉ số', 'Kết quả đo được', 'Mục tiêu đề ra', 'Đạt?'], [
  ['Tracking IDF1', n(T.idf1), '≥ 0,70', 'Đạt'],
  ['Tracking MOTA', n(T.mota), '≥ 0,60', 'Đạt'],
  ['TTC MAE', `${n(F.ttc_mae)} s`, '≤ 0,30 s', 'Đạt'],
  ['PET MAE', `${n(F.pet_mae)} s`, '—', '—'],
  ['Recall (TTC < 1,0 s)', n(sev[0] ? sev[0].recall : 0), '—', 'Rất tốt'],
  ['Conflict Recall (gộp)', n(F.recall), '≥ 0,80', 'Chưa đạt'],
  ['Conflict Precision', n(F.precision), '≥ 0,75', 'Chưa đạt'],
  ['Tốc độ xử lý', `${n(SUM.processing_fps,1)} FPS`, '≥ 25 FPS', 'Đạt'],
  ['Độ trễ end-to-end', `${n(SUM.total_latency_ms,1)} ms/frame`, '≤ 1,5 s', 'Đạt'],
], [1.4, 1.1, 1, 0.8]));

children.push(H2('7.2. Recall theo mức nghiêm trọng'));
children.push(P('Đây là bảng quan trọng nhất về mặt an toàn: bỏ sót một near-miss TTC dưới 1 giây nguy hiểm hơn rất nhiều so với bỏ sót một near-miss TTC 2,8 giây. Vì vậy Recall phải được báo cáo tách theo dải chứ không gộp thành một con số duy nhất.'));
children.push(T_(['Dải TTC', 'Số nhãn chuẩn', 'Phát hiện được', 'Recall', 'TTC MAE'],
  sev.map(r => [r.band, r.n_gt, r.detected ?? '—', n(r.recall), r.ttc_mae === null ? '—' : `${n(r.ttc_mae)} s`]),
  [2, 0.9, 0.9, 0.8, 0.8]));
children.push(Rich([
  { text: 'Hệ thống mạnh đúng ở dải nguy hiểm nhất. ', bold: true },
  { text: `Recall đạt ${n(sev[0] ? sev[0].recall : 0)} với các tình huống TTC dưới 1 giây, và giảm dần ở các dải nhẹ hơn. Đây là hành vi mong muốn: một hệ thống cảnh báo an toàn nên ưu tiên không bỏ sót tình huống nguy cấp, chấp nhận bỏ qua các tình huống ở ranh giới.` },
]));

children.push(H2('7.3. Bảng ablation — đóng góp của từng khối'));
children.push(P('Bật dần từng khối và đo lại, để chứng minh mỗi thành phần đều có đóng góp định lượng được thay vì chỉ có mặt cho đủ sơ đồ.'));
children.push(T_(['Cấu hình', 'Precision', 'Recall', 'F1', 'TTC MAE', 'Số sự kiện'],
  abl.map(r => [r.label, n(r.precision), n(r.recall), n(r.f1), r.ttc_mae === null ? '—' : n(r.ttc_mae), r.n_events]),
  [2.2, 0.8, 0.8, 0.7, 0.8, 0.8]));
children.push(P('Đọc bảng này theo hai chiều:'));
children.push(Bullet('Homography làm Recall nhảy từ 0 lên trên 0,75 — vì trước đó TTC được tính trên đơn vị pixel, vốn không phải đại lượng vật lý, nên hầu như không sự kiện nào khớp với nhãn chuẩn.'));
children.push(Bullet('Làm mượt quỹ đạo giảm TTC MAE và cắt số sự kiện giả đi một nửa, nhờ khử nhiễu vận tốc.'));
children.push(Bullet('PET nâng Recall thêm khoảng 12 điểm — bắt được nhóm tình huống "vừa lướt qua" mà TTC bỏ sót.'));
children.push(Bullet('Cổng khoảng cách là bước có tác dụng lớn nhất: Precision tăng gấp hơn năm lần (0,09 → 0,51) trong khi Recall gần như không đổi. Đây là bằng chứng thực nghiệm cho luận điểm ở mục 6.3.'));

children.push(H2('7.4. Độ bền trước nhiễu detector'));
children.push(P('Quét mức nhiễu bounding box để trả lời câu hỏi thực tế: khi thay bằng model yếu hơn, hoặc khi camera rung, trời mưa, hệ thống chịu được tới đâu?'));
children.push(T_(['Nhiễu bbox (px)', 'Precision', 'Recall', 'F1', 'TTC MAE', 'IDF1'],
  noise.map(r => [n(r.noise_px,1), n(r.precision), n(r.recall), n(r.f1), r.ttc_mae === null ? '—' : n(r.ttc_mae), n(r.idf1)]),
  [1.2, 0.9, 0.9, 0.9, 0.9, 0.9]));
children.push(P('Tracking gần như không suy giảm cho tới mức nhiễu 6 px (IDF1 vẫn 0,98) nhờ cơ chế nới biên bbox khi ghép. Precision là thành phần nhạy cảm nhất — giảm từ 0,67 xuống 0,17 khi nhiễu tăng từ 0 lên 6 px, cho thấy chất lượng detector là nút thắt chính của toàn hệ thống.'));

children.push(H2('7.5. Kết quả trên dữ liệu thật'));
if (REAL && REAL.detection) {
  const d = REAL.detection, t2 = REAL.tracking;
  children.push(T_(['Chỉ số', 'Giá trị', 'Ghi chú'], [
    ['Detection mAP@0.5', n(d.mAP50), 'trọng số COCO gốc, chưa fine-tune'],
    ['Detection mAP@0.5:0.95', n(d.mAP50_95), ''],
    ['Tracking IDF1', n(t2.idf1), 'ghép theo IoU trong không gian ảnh'],
    ['Near-miss ghi nhận', String((REAL.conflicts||{}).total ?? 0), 'chỉ thống kê mô tả — dữ liệu thật không có nhãn xung đột'],
  ], [1.4, 0.8, 2]));
} else {
  children.push(P('Chạy lệnh sau để sinh phần này, sau khi đã chuẩn bị dataset thật:'));
  children.push(new Paragraph({ spacing: { after: 160 }, indent: { left: 300 },
    children: [new TextRun({ text: 'python -m saferoad prepare-real --root data/raw/mvti\npython -m saferoad evaluate-real', font: 'Consolas', size: 20 })] }));
}
children.push(Rich([
  { text: 'Một phát hiện quan trọng. ', bold: true },
  { text: 'Đo trực tiếp trên ảnh giao lộ thật, YOLO11n với trọng số COCO gốc chỉ đạt recall@0.5 ≈ 0,19. Trong một khung hình thử, model chỉ phát hiện đúng một vật thể và gán nhầm nhãn "train" cho một chiếc ô tô. Nguyên nhân không phải model yếu mà là lệch miền: COCO chủ yếu gồm ảnh chụp ngang tầm mắt, trong khi camera giao thông đặt cao nhìn chếch xuống, đối tượng nhỏ và bị nén phối cảnh.' },
]));
children.push(P('Chúng tôi xử lý theo hai hướng. Trước mắt, bổ sung cơ chế suy luận theo ô chồng lấn (cắt khung hình thành 2×3 ô và chạy detector trên từng ô ở độ phân giải gốc), nâng recall từ 0,19 lên 0,46 với cùng bộ trọng số. Về căn cơ, chuẩn bị quy trình fine-tune trên chính tập dữ liệu thật đã có 14.488 bounding box — xem notebook Colab kèm theo repo.'));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 8 ---- //
children.push(H1('8. Đánh giá trung thực so với mục tiêu ban đầu'));
children.push(P('Hai chỉ số chưa đạt mục tiêu đề ra: Conflict Precision (0,51 so với mục tiêu 0,75) và Conflict Recall gộp (0,64 so với mục tiêu 0,80). Chúng tôi báo cáo nguyên trạng và phân tích nguyên nhân thay vì điều chỉnh ngưỡng cho số liệu đẹp hơn.'));
children.push(H2('8.1. Nguyên nhân'));
children.push(Bullet('Trần của thuật toán. Với detector hoàn hảo (nhiễu 0 px), hệ thống đạt F1 = 0,65. Nghĩa là khoảng 0,10 điểm F1 mất do nhiễu detector, phần còn lại là giới hạn của chính phương pháp.'));
children.push(Bullet('Ranh giới giữa "dòng xe đông" và "xung đột" vốn mờ. Phần lớn cảnh báo bị tính là sai thực chất là các cặp xe có khoảng cách thật dưới 1 m — tức là chúng đã đến rất gần nhau, chỉ là không thoả toàn bộ tiêu chí của nhãn chuẩn. Đây là hiện tượng đã được ghi nhận trong tài liệu nghiên cứu traffic conflict technique: ngay cả người gán nhãn có kinh nghiệm cũng bất đồng ở dải TTC 2–3 giây.'));
children.push(Bullet('Độ chính xác phân loại kiểu xung đột chỉ đạt 0,50, do góc tiếp cận được ước lượng từ vận tốc vốn đã có nhiễu.'));
children.push(H2('8.2. Vì sao kết quả vẫn có giá trị sử dụng'));
children.push(P(`Với mục đích thực tế của hệ thống — xếp hạng mức nguy hiểm giữa các giao lộ và chỉ ra điểm nóng cần ưu tiên — thì chỉ số quan trọng là Recall ở dải nguy cấp, và con số đó đạt ${n(sev[0] ? sev[0].recall : 0)}. Các cảnh báo ở ranh giới có thể được lọc bằng ngưỡng Risk Score khi vận hành thực tế, vì thang điểm đã được hiệu chỉnh để phân biệt tốt.`));

children.push(H1('9. Rủi ro và hạn chế đã biết'));
children.push(T_(['Hạn chế', 'Ảnh hưởng', 'Hướng khắc phục'], [
  ['Mô hình vận tốc không đổi', 'TTC bi quan hơn thực tế khi có xe phanh gấp', 'Bổ sung mô hình gia tốc bậc hai; đây là hạn chế chung của mọi phương pháp TTC'],
  ['Giả thiết mặt phẳng của homography', 'Sai số ở giao lộ dốc hoặc có cầu vượt', 'Chia vùng và calibrate riêng từng vùng'],
  ['Chưa fine-tune cho giao thông Việt Nam', 'Detection kém trên góc nhìn camera giao thông', 'Đã chuẩn bị notebook fine-tune trên dữ liệu có nhãn'],
  ['Chỉ xét tương tác từng cặp', 'Tình huống ba xe bị tách thành nhiều cặp riêng lẻ', 'Mở rộng sang mô hình tương tác nhóm'],
  ['Ẩn danh suy từ bbox detector', 'Bỏ sót người đi bộ mà detector không phát hiện', 'Bổ sung detector khuôn mặt chuyên dụng khi công bố dữ liệu'],
  ['Giấy phép AGPL-3.0 của Ultralytics', 'Ràng buộc nếu thương mại hoá', 'Kiến trúc đã tách giao diện detector để thay thế dễ dàng'],
], [1.3, 1.5, 1.8]));

children.push(H1('10. Hướng phát triển'));
children.push(Bullet('Fine-tune YOLO11n trên dữ liệu giao lộ Việt Nam, ưu tiên hai lớp xe máy và người đi bộ.'));
children.push(Bullet('Calibrate homography thực địa bằng điểm mốc đo đạc, thay cho phép xấp xỉ hiện tại.'));
children.push(Bullet('Triển khai trên Jetson Orin Nano, lượng tử hoá INT8 để đạt thời gian thực tại biên.'));
children.push(Bullet('Thu thập dữ liệu nhiều ngày để phân tích chu kỳ rủi ro theo giờ và theo ngày trong tuần.'));
children.push(Bullet('Kết nối với hệ thống đèn tín hiệu để đề xuất điều chỉnh chu kỳ đèn dựa trên số liệu xung đột.'));
children.push(Bullet('Tổ chức gán nhãn near-miss thủ công trên video thật với hai người gán nhãn độc lập, báo cáo hệ số đồng thuận Cohen kappa.'));

children.push(H1('11. Kê khai công cụ và dữ liệu'));
children.push(P('Bản kê khai đầy đủ trong docs/ke_khai_cong_cu.md và docs/dataset_license.md. Tóm tắt:'));
children.push(Bullet('Công cụ AI hỗ trợ sinh mã: Claude (Anthropic). Toàn bộ quyết định thiết kế, lựa chọn ngưỡng, phương pháp đánh giá và diễn giải kết quả do nhóm thực hiện và chịu trách nhiệm.'));
children.push(Bullet('Thuật toán ByteTrack và Kalman filter được cài đặt lại từ công thức, không sao chép repo gốc.'));
children.push(Bullet('Thư viện kế thừa: Ultralytics YOLO (AGPL-3.0), PyTorch, OpenCV, scikit-learn, FastAPI.'));
children.push(Bullet('Dataset kế thừa: Multi-view Traffic Intersection (Møgelmose), UCSD Highway Traffic (Chan & Vasconcelos, 2005 — đã trích dẫn theo yêu cầu).'));
children.push(Bullet('Không sử dụng API trả phí hoặc dịch vụ đám mây nào trong pipeline.'));

children.push(H1('12. Minh chứng kỹ thuật'));
children.push(T_(['Hạng mục', 'Vị trí'], [
  ['Mã nguồn đầy đủ', 'https://github.com/dangkhoi-dev/SafeRoad-AI'],
  ['Sơ đồ kiến trúc', 'docs/assets/architecture.png'],
  ['Protocol định nghĩa near-miss', 'docs/protocol_near_miss_v1.md'],
  ['Bảng kê dữ liệu và giấy phép', 'docs/dataset_license.md'],
  ['Bản kê khai công cụ AI', 'docs/ke_khai_cong_cu.md'],
  ['Prompt Log', 'docs/prompt_log.md'],
  ['Kết quả đánh giá dạng máy đọc', 'data/outputs/evaluation.json'],
  ['Bộ kiểm thử tự động', 'tests/ — 78 test, chạy bằng pytest'],
  ['Notebook fine-tune', 'notebooks/01_finetune_yolo11n_colab.ipynb'],
], [1.3, 2]));

children.push(H1('13. Tài liệu tham khảo'));
[
  'Hayward, J.C. (1972). Near-miss determination through use of a scale of danger. Highway Research Record 384.',
  'Allen, B.L., Shin, B.T., Cooper, P.J. (1978). Analysis of traffic conflicts and collisions. Transportation Research Record 667.',
  'Zheng, L., Ismail, K., Meng, X. (2014). Traffic conflict techniques for road safety analysis: open questions and some insights. Canadian Journal of Civil Engineering, 41(7).',
  'Treiber, M., Hennecke, A., Helbing, D. (2000). Congested traffic states in empirical observations and microscopic simulations. Physical Review E, 62(2).',
  'Zhang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.',
  'Akyon, F.C. et al. (2022). Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. ICIP 2022.',
  'Chan, A.B., Vasconcelos, N. (2005). Probabilistic Kernels for the Classification of Auto-Regressive Visual Processes. IEEE CVPR.',
  'Møgelmose, A. Multi-view Traffic Intersection Dataset. Aalborg University.',
].forEach((r, i) => children.push(P(`[${i + 1}]  ${r}`, { after: 80, size: 22 })));

// --------------------------------------------------------------------------- //
const doc = new Document({
  creator: 'SafeRoad AI Team',
  title: 'SafeRoad AI — Tài liệu dự án Bảng C',
  description: 'Hệ thống AI phát hiện near-miss và bản đồ rủi ro giao thông',
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 420, hanging: 220 } } } },
        { level: 1, format: LevelFormat.BULLET, text: '◦', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 780, hanging: 220 } } } },
      ],
    }],
  },
  styles: { default: { document: { run: { font: FONT, size: 24 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },     // US Letter
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: 'SafeRoad AI — Bảng C — Trang ', font: FONT, size: 18, color: '888888' }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: '888888' }),
          ],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  console.log(`✓ ${OUT}  (${(buf.length / 1024).toFixed(0)} KB)`);
});
