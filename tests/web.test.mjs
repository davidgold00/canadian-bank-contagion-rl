import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {checkPublished,classifyManifest} from '../src/dashboard/assets/update.js';
import {propagate} from '../src/dashboard/assets/scenario.js';
const old='case-aaaaaaaaaaaaaaaa',fresh='case-bbbbbbbbbbbbbbbb';
const manifest=id=>({schema_version:1,snapshot_id:id,feature_date:'2026-09-04',case_url:`/snapshots/${id}/`});
test('public check only reads a noncached publication manifest',async()=>{
 let calls=[];const unchanged=await checkPublished(old,async(...args)=>{calls.push(args);return {ok:true,json:async()=>manifest(old)}});
 assert.equal(unchanged.state,'current');assert.deepEqual(calls,[['/snapshot-manifest.json',{cache:'no-store'}]]);
 const updated=await checkPublished(old,async()=>({ok:true,json:async()=>manifest(fresh)}));assert.equal(updated.state,'new');
});
test('malformed/unavailable/offline responses cannot report success',async()=>{
 for(const payload of [{},manifest(old).case_url,{...manifest(fresh),case_url:'https://evil.example'},null])assert.throws(()=>classifyManifest(old,payload));
 await assert.rejects(checkPublished(old,async()=>({ok:false})),/unavailable/);
 await assert.rejects(checkPublished(old,async()=>({ok:true,json:async()=>{throw new SyntaxError()}})),/malformed/);
 await assert.rejects(checkPublished(old,async()=>{throw new TypeError('offline')}),/offline/);
});
test('scenario recurrence agrees with saved Python output and saturation',async()=>{
 const c=JSON.parse(await readFile(new URL('../artifacts/current/case.json',import.meta.url),'utf8'));const d=c.scenarios;
 const path=propagate(d.matrix,d.presets[d.default]);
 for(let i=0;i<6;i++)for(let j=0;j<6;j++)assert.ok(Math.abs(path[i][j]-d.default_path[i][j])<1e-10);
 const uniform=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>i===j?0:.2));
 for(const initial of Object.values(d.presets))for(const severity of [.5,1,1.5]){
   const run=propagate(uniform,initial.map(x=>x*severity));
   for(let i=1;i<run.length;i++)if(Math.max(...run[i])<100)assert.ok(Math.abs(run[i].reduce((a,b)=>a+b)-1.15*run[i-1].reduce((a,b)=>a+b))<1e-8);
 }
 assert.deepEqual(propagate(uniform,d.presets['Liquidity squeeze'].map(x=>1.5*x)).at(-1),[100,100,100,100,100,100]);
});
