const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + __dirname + '/진행정리_할일_1121.html', { waitUntil: 'networkidle' });
  await p.pdf({ path: __dirname + '/진행정리_할일_1121.pdf', format: 'A4', printBackground: true, margin: { top: '16mm', bottom: '16mm', left: '16mm', right: '16mm' } });
  await b.close(); })();
