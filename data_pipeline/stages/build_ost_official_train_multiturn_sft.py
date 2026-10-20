#!/usr/bin/env python3
"""Convert the released, scene-disjoint OST train split to causal SFT rows."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
from data_pipeline.pure_vlm_utils import PARQUET_ENCODING,canonical,sha256
from data_pipeline.stages.compose_packet_bearing_sft import content_key
F="iclr_spatial.ost_official_train_multiturn_sft.v2"
def main():
 p=argparse.ArgumentParser();p.add_argument("--train-json",type=Path,required=True);p.add_argument("--media-ready",type=Path,required=True);p.add_argument("--test-annotation",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 ready=json.load(open(a.media_ready));scenes=json.load(open(a.train_json));test=json.load(open(a.test_annotation));root=Path(ready["image_root"])
 if not(ready.get("format")=="iclr_spatial.ost_official_train_media_ready.v1" and ready.get("ok") is True and ready.get("scenes")==6390 and ready.get("turns")==47096 and ready.get("unique_images")==206473 and ready.get("missing_images")==0):raise ValueError("media")
 test_ids={str(x["scan_id"]) for x in test};train_ids={str(x["scan_id"]) for x in scenes}
 if len(train_ids)!=6390 or train_ids&test_ids:raise ValueError("scene isolation")
 rows=[];types=Counter();total_turns=0;supervised_turns=0;context_turns=0;media=set();media_refs=0
 for scene in scenes:
  messages=[];ordinals=[];assistant_ordinal=0
  for turn in scene["user_message"]:
   images=[]
   for rel in turn.get("image_paths",[]):
    path=(root/str(rel)).resolve()
    if not path.is_file():raise ValueError(f"missing {path}")
    images.append({"type":"image","image":str(path)});media.add(str(path));media_refs+=1
   messages.append({"role":"user","content":[{"type":"text","text":str(scene["system_prompt"])}]+images+[{"type":"text","text":str(turn["prompt"])}]})
   messages.append({"role":"assistant","content":[{"type":"text","text":str(turn["answer"])}]})
   typ=str(turn.get("type") or "None");types[typ]+=1;total_turns+=1
   if typ!="None":ordinals.append(assistant_ordinal);supervised_turns+=1
   else:context_turns+=1
   assistant_ordinal+=1
  cert={"format":F,"scan_id":scene["scan_id"],"source_split":"official_ost_train","source_license":"CC-BY-NC-4.0","turns":assistant_ordinal,"supervised_turns":len(ordinals),"unsupervised_observation_turns":assistant_ordinal-len(ordinals),"causal_multi_round":True,"system_prompt_repeated_each_user_turn":True,"matches_official_qwen_inference_layout":True,"future_frames_visible":False,"test_scene_overlap":False,"test_fields_read_for_isolation":["scan_id"],"scientific_question_id":"SQ2"};uid="ost-train-scene-"+hashlib.sha256(str(scene["scan_id"]).encode()).hexdigest()[:32]
  rows.append({"uid":uid,"source_uid":str(scene["scan_id"]),"source_group_sha256":hashlib.sha256(str(scene["scan_id"]).encode()).hexdigest(),"component":"official_ost_train_multiturn_scene","target_contract":"causal_multiturn_mixed_spatial_answer","messages":messages,"num_images":sum(len(t.get("image_paths",[])) for t in scene["user_message"]),"num_turns":assistant_ordinal,"sample_loss_weight":1.0,"supervised_assistant_ordinals":ordinals,"ost_official_train_certificate":canonical(cert)})
 if len(rows)!=6390 or total_turns!=47096 or supervised_turns!=45006 or context_turns!=2090 or len(media)!=206473:raise ValueError("counts")
 if len({x["uid"] for x in rows})!=len(rows) or len({content_key(x) for x in rows})!=len(rows):raise ValueError("duplicates")
 a.output_dir.mkdir(parents=True);train=a.output_dir/"train.parquet";pd.DataFrame.from_records(rows).to_parquet(train,index=False,**PARQUET_ENCODING);h=sha256(train);r={"format":F,"ok":True,"rows":len(rows),"train":str(train.resolve()),"train_sha256":h,"scenes":len(train_ids),"total_turns":total_turns,"supervised_turns":supervised_turns,"unsupervised_observation_turns":context_turns,"type_counts":dict(sorted(types.items())),"media_references":media_refs,"unique_images":len(media),"missing_media":0,"scene_overlap_with_test":0,"official_train_split":True,"source_license":"CC-BY-NC-4.0","causal_multi_round":True,"system_prompt_repeated_each_user_turn":True,"matches_official_qwen_inference_layout":True,"future_frames_visible":False,"each_observation_once_per_dialogue":True,"single_turn_flattening":False,"teacher_used_for_targets":False,"benchmark_test_fields_read_for_isolation":["scan_id"],"scientific_question_id":"SQ2","falsifiable_prediction":"official online multi-round supervision improves all OST formats beyond approximate source data while P anchors preserve OVO-S and UCS"};(a.output_dir/"report.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");(a.output_dir/"DATA_READY").write_text("ok\n");print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
