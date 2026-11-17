const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + __dirname + '/발표해설서1118.html', { waitUntil: 'networkidle' });
  await p.pdf({ path: __dirname + '/4팀_발표해설서_1118.pdf', format: 'A4', printBackground: true, margin: { top: '16mm', bottom: '16mm', left: '16mm', right: '16mm' } });
  await b.close(); })();
