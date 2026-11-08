import fs from 'node:fs';
import assert from 'node:assert/strict';
import {resizeRGB} from '../preprocess.js';
const source=fs.readFileSync(new URL('../app.js',import.meta.url),'utf8');
const methods=source.slice(source.indexOf('function roundEven'),source.indexOf('function drawSegmentation'));
const {cropToMask}=new Function('resizeRGB',`${methods}; return {cropToMask};`)(resizeRGB);
const rows=JSON.parse(fs.readFileSync(process.argv[2]));
for(const [i,row] of rows.entries()) {
 const data=k=>new Uint8Array(Buffer.from(row[k],'base64'));
 const rgb=data('rgb'), mask=data('mask'), rgba=new Uint8ClampedArray(640*480*4);
 for(let j=0;j<640*480;j++){rgba[j*4]=rgb[j*3];rgba[j*4+1]=rgb[j*3+1];rgba[j*4+2]=rgb[j*3+2];rgba[j*4+3]=255;}
 for(const [label,actual,expected] of [['crop',cropToMask(rgb,mask),data('crop')],['full',resizeRGB(rgba,640,480),data('full')]]) {
  let changed=0,max=0; for(let k=0;k<actual.length;k++){const d=Math.abs(actual[k]-expected[k]); if(d)changed++; max=Math.max(max,d);}
  assert.ok(max<=1,`${label} ${i}: ${changed} differing bytes, max difference ${max}`);
  fs.writeFileSync(new URL(`../../browser-parity-${i}-${label}.bin`,import.meta.url),actual);
  console.log(`${label} case ${i+1}: ${changed} differing bytes, max difference ${max}.`);
 }
 console.log(`Preprocessing case ${i+1} passed.`);
}
