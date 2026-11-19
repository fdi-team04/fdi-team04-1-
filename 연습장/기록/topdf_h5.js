const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + __dirname + '/H5_논란형_보고서용.html', { waitUntil: 'networkidle' });
  await p.pdf({ path: __dirname + '/H5_논란형_보고서용.pdf', format: 'A4', printBackground: true, margin: { top: '16mm', bottom: '16mm', left: '16mm', right: '16mm' } });
  await b.close(); })();
