// Keep experiment pages inside the local Studio navigation shell.
(() => {
 const routes={cloth:{url:'/projects/cloth/',title:'Cloth workcell'},learned:{url:'/projects/cloth/latent.html',title:'Learned adapters'}};
 const pane=document.createElement('section');pane.id='embedded-experiment';pane.hidden=true;
 pane.innerHTML='<iframe id="experiment-frame" title="Experiment workspace"></iframe>';
 document.querySelector('.application main').append(pane);
 const frame=pane.querySelector('iframe');
 const links=[...document.querySelectorAll('.sidebar a')].filter(a=>a.getAttribute('href')?.startsWith('/projects/cloth/'));
 const show=name=>{
  if(!routes[name])return;
  for(const id of ['workspace-view','tasks-view','analytics-view'])document.getElementById(id).hidden=true;
  if(window.RelayTasks)window.RelayTasks.hide();
  pane.hidden=false;document.body.classList.add('experiment-active');
  document.querySelectorAll('.sidebar .nav-item').forEach(a=>{a.classList.toggle('selected',a.dataset.experiment===name);a.setAttribute('aria-current',a.dataset.experiment===name?'page':'false')});
  document.getElementById('breadcrumb-name').textContent=routes[name].title;
  if(frame.dataset.page!==name){frame.dataset.page=name;frame.src=routes[name].url}
  history.replaceState(null,'',`#${name}`);window.scrollTo({top:0});
 };
 links.forEach(a=>{const name=a.getAttribute('href').includes('latent')?'learned':'cloth';a.dataset.experiment=name;a.removeAttribute('target');a.querySelector('.external')?.remove();a.onclick=event=>{event.preventDefault();show(name)}});
 document.querySelectorAll('.sidebar button,.mobile-view-nav button').forEach(b=>b.addEventListener('click',()=>{pane.hidden=true;document.body.classList.remove('experiment-active');document.getElementById('breadcrumb-name').textContent=(typeof state!=='undefined'&&state.workspace?.name)||'Workspace';links.forEach(a=>{a.classList.remove('selected');a.removeAttribute('aria-current')})},true));
 frame.onload=()=>{const doc=frame.contentDocument;if(!doc)return;const css=doc.createElement('link');css.rel='stylesheet';css.href='/projects/studio/embedded.css?v=3';doc.head.append(css);frame.contentWindow.scrollTo(0,0)};
 window.addEventListener('hashchange',()=>{const name=location.hash.slice(1);if(routes[name])show(name)});
 if(routes[location.hash.slice(1)]){const initial=location.hash.slice(1);const timer=setInterval(()=>{if(document.querySelector('body.auth-pending'))return;clearInterval(timer);show(initial)},100)}
 const simplify=()=>{const title=document.querySelector('.tk-heading h1');if(title&&title.textContent!=='Tasks')title.textContent='Tasks';const subtitle=document.querySelector('.tk-heading p');if(subtitle&&subtitle.textContent!=='Assign work and track the agents handling it.')subtitle.textContent='Assign work and track the agents handling it.'};
 new MutationObserver(simplify).observe(document.getElementById('tasks-view'),{childList:true,subtree:true});simplify();
})();
