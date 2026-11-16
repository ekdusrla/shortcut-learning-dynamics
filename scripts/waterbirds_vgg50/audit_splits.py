"""Read-only source/legacy-split audit; writes only the requested NEW report."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.waterbirds_vgg50.content_split import inventory,legacy_scenes,leakage_report,counts,assign_places


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.root.resolve()):raise ValueError('Audit report must be outside source dataset')
    base,assets,cub,scenes=inventory(args.root,42)
    legacy=legacy_scenes(args.root,base,42)
    report=dict(root=str(args.root.resolve()),original_counts=counts(base),
        cub_files=len(base),scene_files=sum(len(rows) for rows in scenes.values()),
        cub_duplicate_content={h:rows for h,rows in cub.items() if len(rows)>1},
        places_duplicate_content={h:rows for h,rows in scenes.items() if len(rows)>1},
        legacy_split_leakage=leakage_report(legacy,assets),
        legacy_usage_by_correlation={str(c):{mode:leakage_report(assign_places(legacy,c,42,.5),assets,mode) for mode in ('factual','runtime')} for c in (.95,.80,.65)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(cub_files=report['cub_files'],scene_files=report['scene_files'],
                         legacy_split_leakage=report['legacy_split_leakage']['by_domain']),indent=2))

if __name__=='__main__':main()
