export function classifyManifest(current, next) {
  if (!next || next.schema_version !== 1 || typeof next.snapshot_id !== 'string' || !/^case-[a-f0-9]{16}$/.test(next.snapshot_id) || !/^\d{4}-\d{2}-\d{2}$/.test(next.feature_date) || next.case_url !== `/snapshots/${next.snapshot_id}/`) throw new Error('The publication manifest is malformed.');
  return next.snapshot_id === current ? 'current' : 'new';
}
export async function checkPublished(current, fetcher = fetch) {
  const response = await fetcher('/snapshot-manifest.json', {cache:'no-store'});
  if (!response.ok) throw new Error('The publication manifest is unavailable. Your current case is unchanged.');
  let next;
  try { next = await response.json(); } catch { throw new Error('The publication manifest is malformed. Your current case is unchanged.'); }
  return {state:classifyManifest(current,next), next};
}
if (typeof document !== 'undefined') {
  const button=document.querySelector('#check-update'),status=document.querySelector('#update-status'),link=document.querySelector('#open-update');
  if(button) button.addEventListener('click',async()=>{
    button.disabled=true;status.textContent='Checking the published case…';link.hidden=true;
    try { const result=await checkPublished(document.body.dataset.snapshot);
      if(result.state==='current') status.textContent='You are viewing the latest published case. No download or training job was started.';
      else {status.textContent=`A different published case is available, with features through ${result.next.feature_date}. Your current case stays open.`;link.href=result.next.case_url;link.hidden=false;}
    } catch(error) {status.textContent=navigator.onLine===false?'You appear to be offline. Your current case is unchanged.':error.message;}
    finally {button.disabled=false;}
  });
  const toggle=document.querySelector('#nav-toggle'),nav=document.querySelector('#nav-links');
  toggle?.addEventListener('click',()=>{const open=toggle.getAttribute('aria-expanded')!=='true';toggle.setAttribute('aria-expanded',String(open));nav.classList.toggle('open',open);});
  document.querySelector('#print-brief')?.addEventListener('click',()=>window.print());
}
