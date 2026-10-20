#!/usr/bin/env python3
"""Compose released OST multi-round training with P-only preservation data."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from data_pipeline.pure_vlm_utils import PARQUET_ENCODING,builtin,sha256
from data_pipeline.stages.compose_packet_bearing_sft import absolute_media,content_key
F="iclr_spatial.ost_official_train_release.v2";B="iclr_spatial.workfirst_multiview_preservation.v1";O="iclr_spatial.ost_official_train_multiturn_sft.v2"
def load(p):return [{str(k):builtin(v) for k,v in x.items()} for x in pd.read_parquet(p).to_dict("records")]
def main():
 p=argparse.ArgumentParser();p.add_argument("--base-dir",type=Path,required=True);p.add_argument("--ost-dir",type=Path,required=True);p.add_argument("--ost-mass-ratio-to-parent",type=float,choices=(.25,.5,1.0,2.0),required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 bt=a.base_dir/"train.parquet";ot=a.ost_dir/"train.parquet";br=json.load(open(a.base_dir/"report.json"));orr=json.load(open(a.ost_dir/"report.json"))
 if not(br.get("format")==B and br.get("ok") is True and br.get("rows")==33180 and br.get("train_sha256")==sha256(bt) and br.get("benchmark_inputs_read") is False and br.get("benchmark_predictions_or_scores_read") is False and (a.base_dir/"DATA_READY").is_file()):raise ValueError("base")
 if not(orr.get("format")==O and orr.get("ok") is True and orr.get("rows")==6390 and orr.get("scenes")==6390 and orr.get("total_turns")==47096 and orr.get("supervised_turns")==45006 and orr.get("unsupervised_observation_turns")==2090 and orr.get("train_sha256")==sha256(ot) and orr.get("scene_overlap_with_test")==0 and orr.get("official_train_split") is True and orr.get("causal_multi_round") is True and orr.get("system_prompt_repeated_each_user_turn") is True and orr.get("matches_official_qwen_inference_layout") is True and orr.get("future_frames_visible") is False and orr.get("single_turn_flattening") is False and (a.ost_dir/"DATA_READY").is_file()):raise ValueError("ost")
 base=[absolute_media(x,bt.parent) for x in load(bt)];ost_source=[absolute_media(x,ot.parent) for x in load(ot)];seen={content_key(x) for x in base}
 if len(seen)!=len(base):raise ValueError("base dup")
 turns={};ost=[];dropped_context_only=0
 for x in ost_source:
  if content_key(x) in seen:raise ValueError("overlap")
  n=len(x.get("supervised_assistant_ordinals") or [])
  if n<1:dropped_context_only+=1;continue
  seen.add(content_key(x));x["workfirst_family"]="OST";ost.append(x)
  turns[str(x["uid"])]=n
 parent_mass=sum(float(x["sample_loss_weight"]) for x in base);ost_mass=a.ost_mass_ratio_to_parent*parent_mass;per=ost_mass/sum(turns.values());scale=(len(base)+len(ost))/(parent_mass+ost_mass);out=[]
 for x in base:y=dict(x);y["sample_loss_weight"]=float(y["sample_loss_weight"])*scale;out.append(y)
 for x in ost:y=dict(x);y["sample_loss_weight"]=per*turns[str(y["uid"])]*scale;out.append(y)
 mean=sum(float(x["sample_loss_weight"]) for x in out)/len(out);pm=sum(float(x["sample_loss_weight"]) for x in out if str(x.get("workfirst_family") or "")!="OST");om=sum(float(x["sample_loss_weight"]) for x in out if str(x.get("workfirst_family") or "")=="OST")
 if abs(mean-1)>1e-9 or abs(om/pm-a.ost_mass_ratio_to_parent)>1e-8:raise ValueError("mass drift")
 a.output_dir.mkdir(parents=True);train=a.output_dir/"train.parquet";pd.DataFrame.from_records(out).to_parquet(train,index=False,**PARQUET_ENCODING);h=sha256(train);r={"format":F,"ok":True,"rows":len(out),"train":str(train.resolve()),"train_sha256":h,"base_rows":len(base),"ost_source_scenes":len(ost_source),"ost_scene_rows":len(ost),"dropped_context_only_scenes":dropped_context_only,"ost_supervised_turns":sum(turns.values()),"ost_mass_ratio_to_parent":a.ost_mass_ratio_to_parent,"sample_loss_weight_mean":mean,"official_ost_train_split":True,"scene_overlap_with_test":0,"source_license":"CC-BY-NC-4.0","causal_multi_round":True,"future_frames_visible":False,"single_turn_flattening":False,"letter_targets_preserved":True,"release_starts_from_official_base":True,"one_stage_training":True,"checkpoint_continuation":False,"weighted_model_fusion":False,"benchmark_test_fields_read_for_isolation":["scan_id"],"scientific_question_id":"SQ2","falsifiable_prediction":"released in-domain online trajectories raise complete OST toward 58 while P-only preservation retains OVO-S/UCS"};(a.output_dir/"report.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");(a.output_dir/"DATA_READY").write_text("ok\n");print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
