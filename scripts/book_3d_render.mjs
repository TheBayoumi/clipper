import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';

const root=process.cwd();
const output='artifacts/book_3d';
const frames=path.join(output,'frames');
fs.mkdirSync(frames,{recursive:true});
const server=http.createServer((req,res)=>{
 const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
 const filename=path.resolve(root,'.'+pathname);
 if(!filename.startsWith(root+'/')){res.writeHead(403);res.end();return;}
 fs.readFile(filename,(err,bytes)=>{
  if(err){res.writeHead(404);res.end();return;}
  const types={'.html':'text/html','.js':'text/javascript','.json':'application/json',
    '.glb':'model/gltf-binary','.wasm':'application/wasm'};
  res.writeHead(200,{'Content-Type':types[path.extname(filename)]||'application/octet-stream'});
  res.end(bytes);
 });
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({headless:true,args:[
 '--enable-webgl','--use-gl=angle','--use-angle=swiftshader',
 '--enable-unsafe-swiftshader','--disable-dev-shm-usage','--disable-gpu-sandbox']});
try{
 const page=await browser.newPage({viewport:{width:720,height:1280},deviceScaleFactor:1});
 page.on('pageerror',err=>console.error('PAGE ERROR',err.message));
 page.on('console',message=>{if(message.type()==='error')console.error(message.text())});
 const address='http://127.0.0.1:'+server.address().port+'/scripts/book_3d_scene.html';
 await page.goto(address,{waitUntil:'load',timeout:60000});
 await page.waitForFunction('window.__ready === true',null,{timeout:60000});
 const evidence=await page.evaluate(()=>window.__evidence);
 console.log('ACTOR RIG EVIDENCE',JSON.stringify(evidence));
 const sample=[];
 for(let i=0;i<288;i++){
   const info=await page.evaluate(t=>window.renderAt(t),i/24);
   if(i%48===0){sample.push(info);console.log('Recorded actor frame',i,JSON.stringify(info))}
   await page.screenshot({path:path.join(frames,'frame-'+String(i).padStart(4,'0')+'.jpg'),
     type:'jpeg',quality:87});
 }
 const video=path.join(output,'character-acting-review-12s.mp4');
 const audio=path.join(output,'source','voice.wav');
 const args=['-hide_banner','-loglevel','error','-y','-framerate','24',
 '-i',path.join(frames,'frame-%04d.jpg'),'-i',audio,
 '-vf','scale=1080:1920:flags=lanczos,format=yuv420p','-t','12',
 '-af','afade=t=out:st=11.81:d=0.17',
 '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
 '-c:a','aac','-b:a','160k','-movflags','+faststart',video];
 const encoding=spawnSync('ffmpeg',args,{encoding:'utf8',timeout:600000});
 if(encoding.status!==0)throw Error('VIDEO ENCODE FAILED '+encoding.stderr.slice(-1000));
 const decode=spawnSync('ffmpeg',['-v','error','-i',video,'-f','null','-'],
  {encoding:'utf8',timeout:240000});
 if(decode.status!==0)throw Error('FULL DECODE FAILED '+decode.stderr.slice(-1000));
 const probe=spawnSync('ffprobe',['-v','error','-show_streams','-show_format',
  '-of','json',video],{encoding:'utf8'});
 if(probe.status!==0)throw Error('ffprobe failed');
 const meta=JSON.parse(probe.stdout);
 if(!meta.streams.some(s=>s.codec_type==='video'&&s.width===1080&&s.height===1920)
 || !meta.streams.some(s=>s.codec_type==='audio'))throw Error('video/audio contract failed');
 const sha=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
 const sourceDir=path.join(output,'assets');
 const assets=Object.fromEntries(fs.readdirSync(sourceDir).map(f=>
   [f,{sha256:sha(path.join(sourceDir,f))}]));
 const manifest={status:'REAL_SKELETAL_MOTION_REVIEW_ONLY',publishEligible:false,
 originalSourceRun:'40047215101',narration:'cached Kokoro from previous run',
 threeMotionClips:sample,rig:evidence,assets,sourceLicense:'MIT',
 video:{sha256:sha(video),duration:Number(meta.format.duration),width:1080,height:1920},
 fullDecode:true,realBodyAnimation:true,lipsynced:false,creativeApproved:false,
 qualityGatesOutstanding:['professional-host-identity','character-shading','true-face-animation',
 'viseme-sync','book-script-timing','human-visual-review','asset-license-verification']};
 fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2));
 console.log('REAL RIGGED VIDEO REVIEW',JSON.stringify(manifest.video));
}finally{
 await browser.close();
 await new Promise(resolve=>server.close(resolve));
}