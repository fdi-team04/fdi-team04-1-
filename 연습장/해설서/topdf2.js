const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  for (const [src, out] of [['해설서.html','4팀_변수판정_완벽해설서.pdf'],['뉴스로드맵.html','4팀_뉴스_가공검증_로드맵.pdf']]) {
    await p.goto('file://' + __dirname + '/' + src, { waitUntil: 'networkidle' });
    await p.pdf({ path: __dirname + '/' + out, format: 'A4', printBackground: true, displayHeaderFooter: true, headerTemplate: '<div></div>',
      footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#888"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
      margin: { top: '16mm', bottom: '16mm', left: '15mm', right: '15mm' } });
  }
  await b.close();
})();
