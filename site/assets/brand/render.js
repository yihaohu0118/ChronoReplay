/* Original cover artwork and deterministic animation. Generated sculpture is
   loaded unchanged; labels, diagrams, and motion are drawn from editable code. */
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const project = new URLSearchParams(location.search).get('project') || document.body.dataset.project || 'ChronoReplay';
const chrono = project === 'ChronoReplay';
const C = chrono
  ? { accent:'#bb4b2c', bright:'#ffac85', pale:'#f0dfd3', dark:'#262b26', secondary:'#a5c8b0' }
  : { accent:'#6352bd', bright:'#c2b4ff', pale:'#e7e1f5', dark:'#262633', secondary:'#a5c9d8' };
const W = 1600, H = 820, period = 12;
canvas.width = W; canvas.height = H;
const sculpture = new Image();
sculpture.src = 'sculpture.png';
const serif = '"Space Grotesk"';
const sans = 'Manrope';
function txt(s,x,y,size=24,color='#202522',weight=500,family=sans) {
  ctx.font = `${weight} ${size}px ${family}`; ctx.fillStyle=color;
  ctx.fillText(s,x,y);
}
function line(x1,y1,x2,y2,color='#d5d5cc',width=1) {
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();
  ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
}
function box(x,y,w,h,r,fill,stroke) {
  ctx.beginPath();ctx.roundRect(x,y,w,h,r);
  if(fill){ctx.fillStyle=fill;ctx.fill();}
  if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}
}
function dot(x,y,r,color){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
function pill(s,x,y,w,active) {
  box(x,y,w,38,19,active?C.bright:'#343a35',active?null:'#526056');
  txt(s,x+17,y+26,18,active?C.dark:'#d3dbd4',600);
}
function arrow(x1,y,x2,color) {
  line(x1,y,x2,y,color,2);
  line(x2-6,y-5,x2,y,color,2);line(x2-6,y+5,x2,y,color,2);
}
function title(s,x,y,size) {
  ctx.font=`700 ${size}px ${serif}`;
  for(let i=4;i>0;i--){
    ctx.strokeStyle=i===4?C.pale:C.accent+'35';ctx.lineWidth=1;
    ctx.strokeText(s,x+i*3,y+i*3);
  }
  ctx.fillStyle=C.accent;ctx.fillText(s,x,y);
}
function header(t) {
  ctx.fillStyle='#f8f7f3';ctx.fillRect(0,0,W,H);
  // Editorial registration marks and a quiet dot field.
  for(let x=1028;x<1540;x+=22)for(let y=280;y<655;y+=22)dot(x,y,.7,'#c9c8c0');
  line(64,53,1536,53);
  txt(chrono?'CHRONOREPLAY / RESEARCH IN MOTION':'ViSS / RESEARCH IN MOTION',64,35,16,'#596258',600,serif);
  ctx.textAlign='right';txt(`yihaohu0118.github.io/${project}/`,1536,35,16,'#596258',500,serif);ctx.textAlign='left';
  if(chrono){
    title('ChronoReplay',60,188,147);
    txt('One history. Multiple spatial abilities.',69,248,31,'#202522',500,serif);
    txt('CAUSAL REPLAY',1252,143,21,C.accent,600,serif);
    txt('NATIVE TARGETS',1252,174,21,C.accent,600,serif);
    txt('BALANCED LEARNING',1252,205,21,C.accent,600,serif);
  }else{
    title('ViSS',58,213,206);
    txt('Supervising',560,133,43,'#202522',500,serif);
    txt('the visible state.',560,184,43,'#202522',500,serif);
    txt('Learn what matters. Look where it matters.',69,263,31,'#202522',500,serif);
    txt('REPRESENTATION',1252,143,21,C.accent,600,serif);
    txt('×',1252,174,25,C.accent,500,serif);
    txt('VIEW SELECTION',1252,205,21,C.accent,600,serif);
  }
  const cy=480, cx=1278, radius=213;
  ctx.save();ctx.translate(cx,cy);
  for(let i=0;i<60;i++){
    let a=i*Math.PI/30;
    let r=radius-(i%5===0?12:5);
    line(Math.cos(a)*r,Math.sin(a)*r,Math.cos(a)*radius,Math.sin(a)*radius,'#cfcbbf',1);
  }
  const a=t/period*Math.PI*2-Math.PI/2;
  dot(Math.cos(a)*radius,Math.sin(a)*radius,5,C.accent);
  ctx.restore();
  const bounce=4*Math.sin(t/period*Math.PI*2);
  const scale=Math.min(492/sculpture.width,414/sculpture.height);
  const sw=sculpture.width*scale,sh=sculpture.height*scale;
  ctx.drawImage(sculpture,1278-sw/2,477-sh/2+bounce,sw,sh);
  ctx.textAlign='center';txt(chrono?'REPLAY THE PAST. LEARN THE SPACE.':'SAME MODEL. A DIFFERENT VIEW.',1278,687,16,C.accent,600,serif);ctx.textAlign='left';
}
function room(x,y,w,h,index,active) {
  box(x,y,w,h,9,active?'#454e42':'#333a34',active?C.bright:'#586052');
  const mid=x+w*.53,top=y+23,bottom=y+h-20;
  line(x+10,top+14,mid,top,'#82917c',1.2);line(mid,top,x+w-10,top+14,'#82917c',1.2);
  line(mid,top,mid,bottom-14,'#82917c',1.2);
  line(x+10,top+14,x+10,bottom-12,'#82917c',1.2);
  line(x+w-10,top+14,x+w-10,bottom-12,'#82917c',1.2);
  line(x+10,bottom-12,mid,bottom+3,'#82917c',1.2);
  line(mid,bottom+3,x+w-10,bottom-12,'#82917c',1.2);
  const ox=x+18+index*10;
  box(ox,y+53,18,18,3,active?C.bright:'#a8b5a1');
  txt(`t${index+1}`,x+9,y+19,15,active?C.bright:'#a1afa2',600,serif);
}
function chronoScene(t) {
  const phase=Math.min(2,Math.floor(t/4)),p=(t%4)/4;
  box(64,300,915,381,16,C.dark);
  txt('OBSERVATION HISTORY',90,335,16,'#bdc7bb',600,serif);
  txt('CAUSAL BOUNDARIES',743,335,16,'#bdc7bb',600,serif);
  const active=Math.min(5,Math.floor((phase===2?1:p)*6));
  for(let i=0;i<6;i++){
    ctx.globalAlpha=phase===0&&i>active?.2:1;
    room(90+i*144,354,126,111,i,phase<2&&i===active);
    ctx.globalAlpha=1;
    if(i<5)arrow(220+i*144,410,230+i*144,'#a7b6a3');
  }
  line(96,486,943,486,'#66755e',2);
  const cursorX=96+(phase===2?1:p)*847;
  dot(cursorX,486,6,C.bright);
  let labels=['Categorical','Metric','Temporal'];
  let subs=['Which side?','How far?','When?'];
  for(let i=0;i<3;i++){
    const x=90+i*290,lit=phase===2;
    box(x,508,268,65,9,lit?'#404a3d':'#30372f',lit?C.bright:'#596551');
    txt(labels[i],x+17,535,20,lit?C.bright:'#d3dccf',600,serif);
    txt(subs[i],x+17,559,17,'#b1bfa9',500);
  }
  const captions=['Observe in chronological order.','Replay each target at its causal boundary.','Balance complementary supervision families.'];
  txt(captions[phase],90,618,24,'#fff4e9',500,serif);
  const labels2=['01  OBSERVE','02  REPLAY','03  LEARN'];
  for(let i=0;i<3;i++) {
    txt(labels2[i],90+i*290,655,16,i===phase?C.bright:'#91a38a',600,serif);
    line(90+i*290,666,350+i*290,666,'#56634f',2);
    if(i===phase)line(90+i*290,666,90+i*290+260*p,666,C.bright,3);
  }
}
function vissScene(t) {
  const phase=Math.min(2,Math.floor(t/4)),p=(t%4)/4;
  box(64,300,915,381,16,C.dark);
  txt('TVS / COMPLEMENTARY CAUSAL VIEWS',90,335,16,'#c7c3d9',600,serif);
  pill('FROZEN MODEL',754,311,196,true);
  const names=['Recent','Event','History'];
  const desc=['Start with recent frames','Acquire when uncertain','Consult on disagreement'];
  const points=[[.80,.86,.92,.98],[.36,.49,.63,.98],[.06,.49,.63,.98]];
  for(let i=0;i<3;i++){
    const y=358+i*75,active=i===phase;
    box(88,y,865,65,9,active?'#434059':'#30303e',active?C.bright:'#555266');
    txt(names[i],107,y+28,22,active?C.bright:'#d6d1e7',600,serif);
    txt(desc[i],107,y+51,15,'#b6b0cc',500);
    line(419,y+33,909,y+33,'#6f6889',2);
    for(let k=0;k<20;k++)line(419+k*24.5,y+28,419+k*24.5,y+38,'#645e77');
    for(const point of points[i])box(419+point*490-8,y+21,16,24,3,active?C.bright:'#8e839e');
    if(active)dot(419+p*490,y+33,5,'#f9f5ff');
  }
  txt('Look back only within the observed video prefix.',90,618,23,'#f3edff',500,serif);
  txt('ON-DEMAND ACCESS',90,654,16,C.bright,600,serif);
  txt('ILLUSTRATIVE VIEW SELECTION',621,654,16,'#a9a0bd',500,serif);
}
function footer() {
  line(64,714,1536,714);
  const a=chrono?
    [['+13.41','OST-Bench points over the official base'],['3 benchmarks','Improved under matched observation access'],['39,561','Sequences in the causal replay pool']]:
    [['86.33%','StreamingBench · ViSS + TVS'],['69.77%','OVO-Bench · ViSS + TVS'],['7,677','Training examples · one epoch']];
  for(let i=0;i<3;i++){
    const x=64+i*505;
    txt(a[i][0],x,760,35,'#202522',600,serif);
    txt(a[i][1],x,789,17,'#687061',500);
    if(i<2)line(x+475,734,x+475,788);
  }
}
window.renderPoster = function(time=11.2) {
  const t=((time%period)+period)%period;
  header(t);chrono?chronoScene(t):vissScene(t);footer();
};
window.posterReady = Promise.all([
  sculpture.decode(),
  document.fonts.load(`700 147px ${serif}`),
  document.fonts.load(`500 24px ${sans}`),
  document.fonts.load(`600 24px ${sans}`)
]).then(()=>{renderPoster(11.2);return true;});
