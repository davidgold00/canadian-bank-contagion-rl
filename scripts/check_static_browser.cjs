/* Isolated browser regression harness. Install playwright externally or provide NODE_PATH.
   BASE_URL defaults to the local scripts/serve_site.py server; QA_OUTPUT controls screenshots. */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const base = process.env.BASE_URL || 'http://127.0.0.1:8765';
  const output = process.env.QA_OUTPUT || '/private/tmp/northern-qa';
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({headless:true});
  const report = {pages:[], scenarios:[], consoleErrors:[]};
  try {
    for (const width of [1440,390]) {
      const page = await browser.newPage({viewport:{width,height:950}, deviceScaleFactor:1});
      page.on('pageerror',error=>report.consoleErrors.push(error.message));
      page.on('requestfailed',r=>console.log('REQUEST FAILED',r.url(),r.failure()));
      for (const route of ['','risk','scenarios','models','decision','performance','research']) {
        const response=await page.goto(`${base}/${route}`, {waitUntil:'domcontentloaded',timeout:60000});
        assert.equal(response.status(),200,route);
        await page.waitForFunction(() => !document.querySelector('.js-plotly-plot') || !!document.querySelector('.js-plotly-plot')._fullLayout);
        const label=route||'overview';
        await page.screenshot({path:path.join(output,`${width}-${label}-top.png`)});
        const summaries=page.locator('details > summary');
        const count=await summaries.count();
        for(let i=0;i<count;i++) {
          // Enter is also a keyboard check on every disclosure control.
          await summaries.nth(i).focus();
          await page.keyboard.press('Enter');
          assert.equal(await summaries.nth(i).evaluate(node=>node.parentElement.open),true);
        }
        // Trigger native layout resize after opening hidden plots, matching the site's handler.
        await page.evaluate(()=>window.dispatchEvent(new Event('resize')));
        await page.screenshot({path:path.join(output,`${width}-${label}-expanded.png`),fullPage:true});
        const state=await page.evaluate(()=>({width:innerWidth, scrollWidth:document.documentElement.scrollWidth,
          title:document.title,charts:document.querySelectorAll('.js-plotly-plot').length,
          open:document.querySelectorAll('details[open]').length,
          badNumbers:document.body.innerText.match(/\b(?:NaN|undefined)\b/g)||[]}));
        assert.ok(state.scrollWidth<=state.width+1,`${label} overflows: ${JSON.stringify(state)}`);
        assert.equal(state.badNumbers.length,0,`${label}: invalid displayed numbers`);
        report.pages.push({route:label,width,...state});
        for (const section of (route==='decision'?['#rebalance','#change-conditions']:route==='models'?['#cvar-strategy','#comparison','#validation']:route==='performance'?['#paper-portfolio','#benchmarks']:route==='research'?['#validation-method','#metric-method']:route==='risk'?['#composite-score']:[])) {
          await page.locator(section).scrollIntoViewIfNeeded();
          await page.screenshot({path:path.join(output,`${width}-${label}-${section.slice(1)}.png`)});
        }
        if(route!=='scenarios') continue;
        for(const name of ['Housing Crisis','Oil Crash','Liquidity Squeeze','Yield Curve Inversion','Global Risk-Off']) {
          await page.selectOption('#scenario-select',name);
          for(const severity of [50,100,150]) {
            await page.locator('#scenario-severity').focus();
            await page.keyboard.press('Home');
            for(let i=50;i<severity;i+=10) await page.keyboard.press('ArrowRight');
            await page.waitForFunction(({name,severity})=>document.querySelector('#scenario-status').textContent===`Updated ${document.querySelector('#scenario-select').selectedOptions[0].text} at ${severity}% severity.` && document.querySelector('#scenario-path-panel .js-plotly-plot').layout.title.text===`${name} (${severity}%): Contagion Propagation`,{name,severity});
            const active=await page.evaluate(()=>({name:document.querySelector('#scenario-select').value,severity:Number(document.querySelector('#scenario-severity').value),
              title:document.querySelector('#scenario-path-panel .js-plotly-plot').layout.title.text,
              rankingTitle:document.querySelector('#scenario-final-panel .js-plotly-plot').layout.title.text,
              values:document.querySelector('#scenario-final-panel .js-plotly-plot').data[0].x,
              response:document.querySelector('#scenario-response-text').textContent,
              leader:document.querySelector('#scenario-bank').textContent,
              rows:Array.from(document.querySelectorAll('.scenario-impact-table tbody tr')).map(r=>[...r.cells].map(c=>c.textContent))}));
            assert.equal(active.name,name);assert.equal(active.severity,severity);
            assert.ok(active.rankingTitle.startsWith(`${name} (${severity}%)`));
            assert.ok(active.response.includes('does not rerun the optimizer'));
            if(name==='Liquidity Squeeze' && severity===150) {
              assert.ok(active.values.every(v=>v===100));
              assert.ok(active.response.includes('no unique terminal leader'));
              assert.ok(active.response.includes('6 of 6 banks'));
            }
            report.scenarios.push({width,...active});
            await page.locator('#scenario-path-panel').scrollIntoViewIfNeeded();
            await page.screenshot({path:path.join(output,`${width}-scenario-${name.replaceAll(' ','-')}-${severity}.png`)});
          }
        }
        await page.getByRole('button',{name:'Reset scenario'}).click();
        await page.waitForFunction(()=>document.querySelector('#scenario-select').value==='Liquidity Squeeze' && document.querySelector('#scenario-severity').value==='100' && document.querySelector('#scenario-path-panel .js-plotly-plot').layout.title.text==='Liquidity Squeeze (100%): Contagion Propagation');
        // Native select keyboard interaction also updates every dependent title.
        await page.locator('#scenario-select').focus();
        await page.keyboard.type('Housing');
        await page.keyboard.press('Tab');
        await page.waitForFunction(()=>document.querySelector('#scenario-path-panel .js-plotly-plot').layout.title.text==='Housing Crisis (100%): Contagion Propagation');
        assert.equal(await page.getByRole('link',{name:'Review current portfolio decision',exact:true}).getAttribute('href'),'/decision');
      }
      await page.close();
    }
    assert.deepEqual(report.consoleErrors,[]);
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify({pages:report.pages.length,scenarioStates:report.scenarios.length,consoleErrors:report.consoleErrors,output}));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
