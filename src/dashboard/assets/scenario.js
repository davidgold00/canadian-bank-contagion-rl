export function propagate(matrix, initial, severity=1, spillover=.45) {
  let s=initial.map(v=>Math.max(0,Math.min(100,v*severity)));const path=[s];
  for(let k=0;k<5;k++){s=s.map((v,j)=>Math.max(0,Math.min(100,.70*v+spillover*matrix.reduce((sum,row,i)=>sum+row[j]*s[i],0))));path.push(s);}
  return path;
}
export function scenarioResult(data,name,severity) {
  if(!Object.hasOwn(data.presets,name)||!Number.isFinite(severity)||severity<.5||severity>1.5)throw new Error('Invalid scenario');
  const initial=data.presets[name],path=propagate(data.matrix,initial,severity),uniform=data.matrix.map((row,i)=>row.map((_,j)=>i===j?0:1/5));
  return {path,uniform:propagate(uniform,initial,severity),noSpill:propagate(data.matrix,initial,severity,0)};
}
const rank=(values,v)=>1+values.filter(x=>Number(x.toFixed(1))>Number(v.toFixed(1))).length;
if(typeof document!=='undefined'){
  const source=document.querySelector('#scenario-data');
  if(source){const data=JSON.parse(source.textContent),params=new URLSearchParams(location.search);
    let name=Object.hasOwn(data.presets,params.get('scenario'))?params.get('scenario'):data.default;
    let severity=Number(params.get('severity')||100);if(!Number.isFinite(severity)||severity<50||severity>150)severity=100;severity=Math.round(severity/10)*10;
    const select=document.querySelector('#scenario-preset'),slider=document.querySelector('#severity');
    if(select){select.value=name;slider.value=severity;}
    function update(writeUrl=false){
      if(select){name=select.value;severity=Number(slider.value);document.querySelector('#severity-label').textContent=severity+'%';}
      const r=scenarioResult(data,name,severity/100),first=r.path[0],final=r.path[5];
      const query=new URLSearchParams({scenario:name,severity:String(severity)});
      if(writeUrl)history.replaceState(null,'',location.pathname+'?'+query.toString());
      const attached=document.querySelector('#attached-scenario');
      if(attached&&params.has('scenario')){attached.hidden=false;attached.textContent=`Supporting sensitivity: ${name}, ${severity}% severity. Initial mean ${(first.reduce((a,b)=>a+b,0)/6).toFixed(1)}; final mean ${(final.reduce((a,b)=>a+b,0)/6).toFixed(1)}. Five abstract steps; no optimizer rerun. URL: ${location.href}`;}
      if(!select)return;
      const set=(id,text)=>document.getElementById(id).textContent=text;
      set('scenario-caption',`${name} · ${severity}% severity`);set('scenario-status',`${name} at ${severity}% severity. Outputs updated.`);
      set('initial-mean',(first.reduce((a,b)=>a+b,0)/6).toFixed(1));set('final-mean',(final.reduce((a,b)=>a+b,0)/6).toFixed(1));const capped=final.filter(v=>v>=100).length;set('cap-count',`${capped} / 6`);
      const tbody=document.querySelector('#scenario-rows');tbody.replaceChildren();
      data.banks.forEach((bank,i)=>{const tr=document.createElement('tr');[bank,first[i].toFixed(1),final[i].toFixed(1),`${rank(first,first[i])} → ${rank(final,final[i])}`,r.uniform[5][i].toFixed(1),`${rank(final,final[i])} → ${rank(r.uniform[5],r.uniform[5][i])}`,r.noSpill[5][i].toFixed(1)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});tbody.append(tr);});
      set('scenario-totals',`Aggregate intensity: ${first.reduce((a,b)=>a+b,0).toFixed(1)} initially → ${final.reduce((a,b)=>a+b,0).toFixed(1)} after five steps. These sums are index units, not monetary losses.`);
      const max=Math.max(...final),leaders=data.banks.filter((b,i)=>Number(final[i].toFixed(1))===Number(max.toFixed(1)));
      set('leader-note',capped===6?'All six banks reach the cap. Saturation removes differentiation.':`Highest terminal modeled intensity at displayed precision: ${leaders.join(', ')}. This ranking is conditional on the assumptions.`);
      const incoming=data.matrix.reduce((sum,row,i)=>sum+row[0]*first[i],0)*.45;
      set('step-example',`Worked first step for ${data.banks[0]}: ${(first[0]*.70).toFixed(2)} retained + ${incoming.toFixed(2)} incoming = ${r.path[1][0].toFixed(2)} after clipping. Net change ${(r.path[1][0]-first[0]).toFixed(2)} is not spillover alone.`);
      const normalized=data.matrix.every(row=>Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-8);
      set('aggregate-explanation',`Without clipping${normalized?'': ' and if every outgoing row were normalized'}, the five-step aggregate multiplier is 1.15⁵ = 2.011. This preset starts at mean ${(first.reduce((a,b)=>a+b,0)/6).toFixed(2)}; the uncapped normalized-row reference is ${(first.reduce((a,b)=>a+b,0)/6*1.15**5).toFixed(2)}. ${capped?'Clipping changes the observed total.':'The network redistributes this configured amplification.'}`);
      const pathNode=document.querySelector('#scenario-paths');pathNode.replaceChildren();r.path.forEach((step,k)=>{const p=document.createElement('p');p.textContent=`Step ${k}: `+data.banks.map((b,i)=>`${b} ${step[i].toFixed(2)}`).join('; ');pathNode.append(p);});
      const link=document.querySelector('#scenario-review');link.href=link.href.split('?')[0]+'?'+query.toString();
    }
    select?.addEventListener('change',()=>update(true));slider?.addEventListener('input',()=>update(true));
    document.querySelector('#scenario-reset')?.addEventListener('click',()=>{select.value=data.default;slider.value=100;update(true);});
    document.querySelector('#copy-scenario')?.addEventListener('click',async()=>{update(true);try{await navigator.clipboard.writeText(location.href);document.querySelector('#scenario-status').textContent='Scenario link copied.';}catch{document.querySelector('#scenario-status').textContent='Copy this URL from the address bar: '+location.href;}});
    update(false);
  }
}
