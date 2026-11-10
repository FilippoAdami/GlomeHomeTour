'use strict';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
let data, active = -1, yaw = 0, pitch = 0, fov = 2 * Math.PI / 3, gl, program, texture, uniforms, dragging;
let viewVersion = 0, wobbleFrame = 0, hintTimer = 0, cueAllowed = false, panoramaMode = 'gaussian';
const radians = x => x * Math.PI / 180;
function element(name, attrs = {}) { const node = document.createElementNS(NS, name); for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value); return node; }
function currentLevel() { return data.levels.find(level => level.name === $('level').value); }
function map(container, level, style, cameras, crop, current, showMeasurements = false) {
  const info = level.styles[style], grid = info.grid, box = crop || [0, 0, grid.width, grid.height];
  container.replaceChildren(); container.style.aspectRatio = current ? '1' : `${box[2]-box[0]} / ${box[3]-box[1]}`;
  const width=box[2]-box[0], height=box[3]-box[1], size=Math.max(width,height);
  const viewBox=current ? [box[0]-(size-width)/2,box[1]-(size-height)/2,size,size] : [box[0],box[1],width,height];
  const svg = element('svg', {viewBox: viewBox.join(' '), role:'group', 'aria-label': current ? 'Current camera minimap' : 'Floor plan camera markers'});
  let layer=svg;
  if(current) {
    const defs=element('defs'), clip=element('clipPath',{id:'minimap-crop',clipPathUnits:'userSpaceOnUse'});
    clip.append(element('rect',{x:box[0],y:box[1],width,height})); defs.append(clip); svg.append(defs);
    layer=element('g',{'clip-path':'url(#minimap-crop)'}); svg.append(layer);
  }
  const image = element('image', {href:info.image, x:0, y:0, width:grid.width, height:grid.height}); layer.append(image);
  if (showMeasurements && !current) {
    const overlay=level.measurements, source=overlay.grid, scale=source.size/grid.size;
    const x=(source.x_min-grid.x_min)/grid.size, y=(grid.z_max-source.z_max)/grid.size;
    const width=source.width*scale, height=source.height*scale;
    layer.append(element('image', {href:overlay.image,
      x, y:grid.height-y-height,
      width, height, 'pointer-events':'none'}));
  }
  for (const camera of cameras) {
    const [x,y] = camera.pixels[style], group = element('g', {class:`camera ${camera.kind === 'photo' ? 'photo' : 'pano'} ${camera === current ? 'current' : ''}`, ...(current ? {} : {tabindex:'0', role:'button', 'aria-label':`${camera.scan}, ${camera.kind}, camera ${camera.number}`})});
    const direction = camera === current && camera.kind === 'panorama' ? yaw : camera.yaw;
    const edges = camera === current && camera.kind === 'panorama' ? [-fov/2,fov/2] : camera.edges;
    if (camera.kind === 'photo' || current) {
      const pts = edges.map(a => {
        const angle = -(direction + a);
        return `${x+90*Math.cos(angle)},${y-90*Math.sin(angle)}`;
      });
      group.append(element('polygon', {points:`${x},${y} ${pts.join(' ')}`, class:'cone'}));
    } else group.append(element('circle', {cx:x, cy:y, r:31, class:'ring'}));
    group.append(element('circle',{cx:x,cy:y,r:38,fill:'transparent'}));
    group.append(element('circle',{cx:x,cy:y,r:21,class:'disc'}));
    const label = element('text',{x,y}); label.textContent=camera.number; group.append(label);
    if (!current) { const open = () => showCamera(data.cameras.indexOf(camera)); group.addEventListener('click',open); group.addEventListener('keydown',e => {if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}}); }
    layer.append(group);
  }
  container.append(svg);
}
function renderPlan() { const level=currentLevel(); map($('plan'),level,$('style').value,data.cameras.filter(c=>c.floor===level.name),null,null,$('measurements').checked); }
function renderMini() { const camera=data.cameras[active], level=currentLevel(), style=$('style').value; map($('minimap'),level,style,[camera],level.styles[style].crops[camera.scan],camera); }
function stopCue() { cueAllowed=false; cancelAnimationFrame(wobbleFrame); wobbleFrame=0; clearTimeout(hintTimer); $('panorama-hint').hidden=true; }
function panoramaSource(camera) { return panoramaMode === 'mesh' ? camera.mesh_image : camera.image; }
function updatePanoramaButton() {
  const target=panoramaMode === 'gaussian' ? 'mesh' : 'Gaussian splat';
  $('panorama-source').textContent=`Show ${target}`;
  $('panorama-source').setAttribute('aria-label', `Switch to ${target} panorama`);
}
function showPlan() { stopCue(); viewVersion++; active=-1; $('plan-menu').hidden=true; $('plan-view').hidden=false; $('image-view').hidden=true; renderPlan(); }
function showCamera(index) {
  stopCue(); const version=++viewVersion;
  active=(index+data.cameras.length)%data.cameras.length; const c=data.cameras[active]; $('level').value=c.floor;
  $('plan-menu').hidden=true; $('plan-view').hidden=true; $('image-view').hidden=false;
  $('minimap-open').setAttribute('aria-label', `Return to floor plan, ${c.scan}, camera ${c.number}`);
  $('photo').alt=`${c.scan}, ${c.floor}, camera ${c.number}`;
  $('photo').hidden=c.kind!=='photo'; $('panorama').hidden=c.kind!=='panorama'; $('pano-controls').hidden=c.kind!=='panorama';
  if(c.kind==='photo') $('photo').src=c.image; else {yaw=c.yaw; pitch=0; fov=radians(120); cueAllowed=true; updatePanoramaButton(); loadPanorama(panoramaSource(c),version);}
  renderMini();
}
function initGL() {
  const canvas=$('panorama'); gl=canvas.getContext('webgl'); if(!gl) throw Error('Your browser cannot display WebGL panoramas.');
  const shader=(type,src)=>{const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
  program=gl.createProgram(); gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec2 p; void main(){gl_Position=vec4(p,0.,1.);}'));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision highp float; uniform sampler2D image; uniform vec2 resolution; uniform float yaw,pitch,fov; const float PI=3.141592653589793; void main(){vec2 uv=(gl_FragCoord.xy/resolution*2.-1.); float t=tan(fov*.5); vec3 forward=vec3(cos(pitch)*cos(yaw),sin(pitch),cos(pitch)*sin(yaw)); vec3 right=vec3(-sin(yaw),0.,cos(yaw)); vec3 up=vec3(-sin(pitch)*cos(yaw),cos(pitch),-sin(pitch)*sin(yaw)); vec3 d=normalize(forward+uv.x*t*right+uv.y*t*resolution.y/resolution.x*up); float u=atan(d.z,d.x)/(2.*PI)+.5; float v=.5-asin(clamp(d.y,-1.,1.))/PI; gl_FragColor=texture2D(image,vec2(u,v));}`));
  gl.linkProgram(program); if(!gl.getProgramParameter(program,gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program)); gl.useProgram(program);
  const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);const p=gl.getAttribLocation(program,'p');gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,2,gl.FLOAT,false,0,0);
  uniforms={resolution:gl.getUniformLocation(program,'resolution'),yaw:gl.getUniformLocation(program,'yaw'),pitch:gl.getUniformLocation(program,'pitch'),fov:gl.getUniformLocation(program,'fov')};
  texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
}
function loadPanorama(src,version) { if(!gl)initGL(); const image=new Image(); image.onload=()=>{if(version!==viewVersion)return;gl.bindTexture(gl.TEXTURE_2D,texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,image);renderPano();if(cueAllowed)startCue();}; image.src=src; }
function startCue() {
  $('panorama-hint').hidden=false;
  hintTimer=setTimeout(()=>{$('panorama-hint').hidden=true;},3000);
  if(window.matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const initialYaw=yaw; let started;
  function frame(time) {
    if(started===undefined)started=time;
    const progress=Math.min((time-started)/1300,1);
    yaw=initialYaw+radians(9)*Math.sin(2*Math.PI*progress)*Math.sin(Math.PI*progress);
    renderPano();
    wobbleFrame=progress<1 ? requestAnimationFrame(frame) : 0;
  }
  wobbleFrame=requestAnimationFrame(frame);
}
function renderPano() { if(active<0 || data.cameras[active].kind!=='panorama'||!gl)return; const canvas=$('panorama'), scale=Math.min(devicePixelRatio,2), width=Math.max(1,Math.round(canvas.clientWidth*scale)), height=Math.max(1,Math.round(canvas.clientHeight*scale));if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}gl.viewport(0,0,width,height);gl.uniform2f(uniforms.resolution,width,height);gl.uniform1f(uniforms.yaw,yaw);gl.uniform1f(uniforms.pitch,pitch);gl.uniform1f(uniforms.fov,fov);gl.drawArrays(gl.TRIANGLE_STRIP,0,4);const [x,y]=data.cameras[active].pixels[$('style').value], cone=$('minimap').querySelector('.current .cone');if(cone){const points=[-fov/2,fov/2].map(a=>`${x+90*Math.cos(-(yaw+a))},${y-90*Math.sin(-(yaw+a))}`);cone.setAttribute('points',`${x},${y} ${points.join(' ')}`);} }
function zoom(delta){stopCue();fov=Math.max(radians(35),Math.min(radians(120),fov+delta));renderPano();}
$('minimap-open').onclick=showPlan;$('previous').onclick=()=>showCamera(active-1);$('next').onclick=()=>showCamera(active+1);
$('level').onchange=()=>{if(active>=0)showPlan();else renderPlan();};$('style').onchange=()=>active<0?renderPlan():renderMini();
$('measurements').onchange=renderPlan;
for(const id of ['plan','minimap-open']) $(id).addEventListener('contextmenu',e=>{e.preventDefault();const menu=$('plan-menu');menu.hidden=false;menu.style.left=`${Math.min(e.clientX,innerWidth-menu.offsetWidth-4)}px`;menu.style.top=`${Math.min(e.clientY,innerHeight-menu.offsetHeight-4)}px`;menu.querySelector(`[data-style="${$('style').value}"]`).focus();});
$('plan-menu').addEventListener('click',e=>{const choice=e.target.closest('[data-style]');if(!choice)return;$('style').value=choice.dataset.style;$('style').dispatchEvent(new Event('change'));$('plan-menu').hidden=true;});
document.addEventListener('click',e=>{if(!$('plan-menu').contains(e.target))$('plan-menu').hidden=true;});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('plan-menu').hidden){$('plan-menu').hidden=true;(active<0?$('style'):$('minimap-open')).focus();}});
$('zoom-in').onclick=()=>zoom(-radians(10));$('zoom-out').onclick=()=>zoom(radians(10));
$('panorama-source').onclick=()=>{if(active<0 || data.cameras[active].kind!=='panorama')return;stopCue();panoramaMode=panoramaMode==='gaussian'?'mesh':'gaussian';updatePanoramaButton();loadPanorama(panoramaSource(data.cameras[active]),++viewVersion);};
$('panorama').addEventListener('pointerdown',e=>{stopCue();dragging={x:e.clientX,y:e.clientY};e.target.setPointerCapture(e.pointerId);});
$('panorama').addEventListener('pointermove',e=>{if(!dragging)return;const k=2*Math.tan(fov/2)/e.target.clientWidth;yaw-=(e.clientX-dragging.x)*k;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-dragging.y)*k));dragging={x:e.clientX,y:e.clientY};renderPano();});
$('panorama').addEventListener('pointerup',()=>dragging=null);$('panorama').addEventListener('pointercancel',()=>dragging=null);
$('panorama').addEventListener('wheel',e=>{e.preventDefault();zoom(Math.sign(e.deltaY)*radians(5));},{passive:false});
$('panorama').addEventListener('keydown',e=>{const step=radians(5);if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-'].includes(e.key))return;stopCue();if(e.key==='ArrowLeft')yaw-=step;else if(e.key==='ArrowRight')yaw+=step;else if(e.key==='ArrowUp')pitch=Math.min(1.5,pitch+step);else if(e.key==='ArrowDown')pitch=Math.max(-1.5,pitch-step);else if(e.key==='+'||e.key==='=')zoom(-step);else zoom(step);e.preventDefault();renderPano();});
window.addEventListener('resize',renderPano);
fetch('tour.json', {cache:'no-store'}).then(r=>{if(!r.ok)throw Error(`Cannot load tour: ${r.status}`);return r.json();}).then(result=>{data=result; $('measurement-tooltip').textContent=data.measurement_warning; for(const level of data.levels){const option=document.createElement('option');option.value=level.name;option.textContent=level.name;$('level').append(option);}renderPlan();}).catch(error=>{document.querySelector('main').textContent=`Unable to load the tour: ${error.message}`;});
