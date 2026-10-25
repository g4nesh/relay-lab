const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const smooth=(a,b,v)=>{const t=clamp((v-a)/(b-a));return t*t*(3-2*t)};
const lerp=(a,b,t)=>a+(b-a)*t;
const chapters=[
 {start:0,title:'Different brains. A common plan.',tag:'01 / ASSIGN THE MISSION',description:'Atlas supplies a red housing. Nova loads a blue insert. Each VLA executes its own robot’s actions.',route:'RELAY → ATLAS + NOVA',message:'“Prepare one component kit.”',status:'TASK OWNERSHIP RECORDED',kind:'assignment'},
 {start:.07,title:'Two VLAs, working in parallel.',tag:'02 / OVERLAP INDEPENDENT WORK',description:'VLA A picks the housing while VLA B picks the insert. Separate work areas mean neither needs to wait.',route:'ATLAS ↔ NOVA',message:'“I’m supplying the housing.”  /  “I’m loading the insert.”',status:'BOTH ROBOTS EXECUTING THEIR OWN POLICIES',kind:'intention'},
 {start:.2,title:'Share the plan. Reserve the space.',tag:'03 / COORDINATE THE SHARED ZONE',description:'Relay grants Atlas access before it enters the transfer zone. Nova finishes loading the insert independently.',route:'RELAY → ATLAS',message:'“Transfer zone reserved. Proceed with the housing.”',status:'RESOURCE GRANTED / ATLAS',kind:'reservation'},
 {start:.44,title:'A handoff starts with a message.',tag:'04 / REPORT READINESS + CONFIRM CLEARANCE',description:'Once the housing is in the tray and Atlas is clear, its agent tells Nova exactly what is ready and where.',route:'ATLAS → NOVA',message:'“Housing 01 is in the transfer tray. The zone is clear.”',status:'HANDOFF_READY / VERIFIED SIMULATION STATE',kind:'handoff'},
 {start:.54,title:'Nova accepts. VLA B takes over.',tag:'05 / ACCEPT THE HANDOFF',description:'Nova’s agent accepts the handoff and receives shared-zone access. Its own VLA performs the pickup.',route:'NOVA → ATLAS',message:'“Handoff accepted. Collecting housing 01.”',status:'RESOURCE GRANTED / NOVA',kind:'handoff'},
 {start:.69,title:'Keep independent work moving.',tag:'06 / OVERLAP THE NEXT TASK',description:'While Nova packs the housing, Atlas prepares the next one. Only access to the shared zone is sequenced.',route:'ATLAS ↔ NOVA',message:'“Preparing the next housing.”  /  “Finishing Kit 01.”',status:'PARALLEL WORK / SEPARATE WORK AREAS',kind:'intention'},
 {start:.9,title:'Two different VLAs. One verified kit.',tag:'07 / VERIFY THE SHARED OUTCOME',description:'Both components are verified. Independent work overlaps; shared access stays ordered. Relay is designed to reduce unnecessary waiting.',route:'RELAY → OPERATOR',message:'“Kit 01 verified. Both required components are present.”',status:'COMPLETED / VERIFIED SIMULATION STATE',kind:'verified'}
];
const atlasKeys=[
 [0,-2.9,1,1.05],[.07,-2.9,1,1.05],[.11,-4.2,-2,1.05],[.135,-4.2,-2,.45],[.16,-4.2,-2,.45],[.185,-4.2,-2,1.05],[.29,0,-.1,1.05],[.32,0,-.1,.45],[.34,0,-.1,.45],[.37,0,-.1,1.05],[.44,-2.25,1,1.05],[.53,-2.25,1,1.05],[.605,-4.2,-2.8,1.05],[.65,-4.2,-2.8,.45],[.67,-4.2,-2.8,.45],[.705,-4.2,-2.8,1.05],[.82,-2.65,-1.5,1.05],[1,-2.65,-1.5,1.05]
];
const novaKeys=[
 [0,2.9,.8,1.05],[.07,2.9,.8,1.05],[.11,4.7,1.9,1.05],[.13,4.7,1.9,.45],[.155,4.7,1.9,.45],[.18,4.7,1.9,1.05],[.235,3.6,-2.1,1.05],[.267,3.6,-2.1,.45],[.28,3.6,-2.1,.45],[.31,3.6,-2.1,1.05],[.36,2.7,.8,1.05],[.54,2.7,.8,1.05],[.61,0,-.1,1.05],[.65,0,-.1,.45],[.67,0,-.1,.45],[.695,0,-.1,1.05],[.75,2.2,-1.2,1.05],[.8,2.95,-2.1,1.05],[.835,2.95,-2.1,.45],[.85,2.95,-2.1,.45],[.88,2.95,-2.1,1.05],[.93,2.9,.8,1.05],[1,2.9,.8,1.05]
];
function sample(keys,t){let a=keys[0],b=keys.at(-1);for(let i=1;i<keys.length;i++){if(t<=keys[i][0]){a=keys[i-1];b=keys[i];break;}}const u=smooth(a[0],b[0],t);return {x:lerp(a[1],b[1],u),y:lerp(a[3],b[3],u),z:lerp(a[2],b[2],u)};}
function stateAt(t){return {t,atlas:sample(atlasKeys,t),nova:sample(novaKeys,t),redOwner:t<.145?'supply':t<.34?'atlas':t<.66?'transfer':t<.85?'nova':'kit',blueOwner:t<.14?'supply':t<.275?'nova':'kit',nextOwner:t<.66?'supply':'atlas',zone:t>=.2&&t<.44?'Atlas':t>=.54&&t<.735?'Nova':null,verified:t>=.9};}

function objectPosition(s,id){
 const held=p=>({x:p.x,y:p.y-.33,z:p.z});
 if(id==='housing-1')return s.redOwner==='atlas'?held(s.atlas):s.redOwner==='nova'?held(s.nova):s.redOwner==='supply'?{x:-4.2,y:.12,z:-2}:s.redOwner==='transfer'?{x:0,y:.12,z:-.1}:{x:2.95,y:.12,z:-2.1};
 if(id==='insert-1')return s.blueOwner==='nova'?held(s.nova):s.blueOwner==='supply'?{x:4.7,y:.12,z:1.9}:{x:3.6,y:.12,z:-2.1};
 return s.nextOwner==='atlas'?held(s.atlas):{x:-4.2,y:.12,z:-2.8};
}
export {chapters,stateAt,objectPosition};
