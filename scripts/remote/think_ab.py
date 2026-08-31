import json,time,urllib.request
URL="http://127.0.0.1:8011/v1/chat/completions"
Q=("Rate this animation step on a letter band A-E for pacing. "
   "Reply with JSON only: {\"band\":\"A\"}. Step 3 of 12 reveals a titled block.")
def run(think):
    body={"model":"moonshotai/Kimi-K2.6","messages":[{"role":"user","content":Q}],
          "max_tokens":2048,"temperature":0.0}
    if think is not None: body["chat_template_kwargs"]={"thinking":think}
    r=urllib.request.Request(URL,data=json.dumps(body).encode(),
                             headers={"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=600)); el=time.time()-t
    u=d["usage"]; txt=d["choices"][0]["message"]["content"] or ""
    return el,u["completion_tokens"],txt.strip()[:60]
for label,th in [("thinking ON (default)",None),("thinking OFF",False)]:
    try:
        el,ct,txt=run(th)
        print(f"  {label:24s} {el:6.1f}s  {ct:5d} completion tok  | {txt}")
    except Exception as e:
        print(f"  {label:24s} ERR {type(e).__name__} {str(e)[:70]}")
