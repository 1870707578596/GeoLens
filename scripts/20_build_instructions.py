# GeoLens Phase 1 —— 指令数据生成（clean / raw 双版本）
# 用法（过滤分析完成后）：
#   python scripts/20_build_instructions.py --labels data/labels.jsonl \
#       --decisions data/filter_decisions.jsonl \
#       --out-clean data/instructions_clean.json --out-raw data/instructions_raw.json
# 逻辑：
#   - raw：全部训练图（train+validation split），答案取 CLIP top1（含噪声，对照组）
#   - clean：仅 keep=True 的图（置信度过滤后，实验组）
#   - 每图随机生成 2-3 条指令：分类 / 二元判断 / 内容描述 / 多标签列举
#   - 输出 LLaMA-Factory sharegpt 视觉格式：{"conversations": [...], "images": [path]}
#   - test split 不进入训练数据（留给评测）

import argparse
import json
import random
from pathlib import Path

# 45 类中文名与描述句（模板图注知识库，替代原计划的外部 caption 数据集，见 DECISIONS D5/D9）
CLASS_ZH = {
    "airplane": ("飞机", "停机坪上整齐排列着多架飞机"),
    "airport": ("机场", "机场的跑道与航站楼清晰可见"),
    "baseball_diamond": ("棒球场", "棒球场呈现标志性的扇形钻石轮廓"),
    "basketball_court": ("篮球场", "篮球场的矩形场地与标线清晰可辨"),
    "beach": ("海滩", "绵长的海岸线与沙滩展现在画面中"),
    "bridge": ("桥梁", "一座桥梁横跨在水面或沟壑之上"),
    "chaparral": ("灌木丛", "大片低矮的灌木植被覆盖着地表"),
    "church": ("教堂", "教堂建筑及其尖顶在建筑群中十分醒目"),
    "circular_farmland": ("环形农田", "中心灌溉式农田呈现规则的圆形图案"),
    "cloud": ("云", "厚重的云层覆盖了大部分画面"),
    "commercial_area": ("商业区", "密集的商业建筑与街区纵横交错"),
    "dense_residential": ("高密度住宅区", "高密度的住宅建筑紧密相连"),
    "desert": ("沙漠", "广阔的沙漠地带几乎不见植被"),
    "forest": ("森林", "茂密的森林覆盖了整片区域"),
    "freeway": ("高速公路", "高速公路笔直地延伸向远方"),
    "golf_course": ("高尔夫球场", "高尔夫球场的草坪与球道呈带状分布"),
    "ground_track_field": ("田径场", "田径场的环形跑道轮廓分明"),
    "harbor": ("港口", "港口码头旁停泊着多艘船只"),
    "industrial_area": ("工业区", "成片的厂房与工业设施分布其间"),
    "intersection": ("十字路口", "道路在此交汇形成十字路口"),
    "island": ("岛屿", "一座岛屿被水体环绕"),
    "lake": ("湖泊", "湖泊的水面占据了画面的主要部分"),
    "meadow": ("草地", "开阔的草地上植被稀疏均匀"),
    "medium_residential": ("中密度住宅区", "中等密度的住宅区房屋排列整齐"),
    "mobile_home_park": ("活动房屋区", "成排的活动房屋紧凑地排列着"),
    "mountain": ("山脉", "连绵的山脉与山谷交错分布"),
    "overpass": ("立交桥", "立交桥的匝道在地面交织缠绕"),
    "palace": ("宫殿", "宫殿建筑群规模宏大、布局对称"),
    "parking_lot": ("停车场", "停车场内车辆整齐地停放着"),
    "railway": ("铁路", "铁路轨道蜿蜒穿过画面"),
    "railway_station": ("火车站", "火车站的站台与轨道清晰可见"),
    "rectangular_farmland": ("矩形农田", "矩形农田地块整齐地拼接在一起"),
    "river": ("河流", "一条河流曲折地穿过区域"),
    "roundabout": ("环岛", "道路环绕圆形岛形成环岛"),
    "runway": ("飞机跑道", "飞机跑道笔直地延伸"),
    "sea_ice": ("海冰", "海面上漂浮着大片的浮冰"),
    "ship": ("船舶", "多艘船舶航行或停泊在水域中"),
    "snowberg": ("雪山", "积雪覆盖的山体反射着亮色"),
    "sparse_residential": ("低密度住宅区", "零星的房屋散布在开阔地带"),
    "stadium": ("体育场", "体育场呈椭圆形的宏大轮廓"),
    "storage_tank": ("储油罐", "成组的圆形储罐密集排列"),
    "tennis_court": ("网球场", "网球场地的蓝色或绿色标线清晰"),
    "terrace": ("梯田", "层层梯田沿山势盘绕而上"),
    "thermal_power_station": ("火力发电站", "发电厂的冷却塔与烟囱设施醒目"),
    "wetland": ("湿地", "湿地水体与植被交错分布"),
}

Q_CLASS = [
    "这张遥感影像中的主要地物是什么？",
    "请识别这张遥感影像中的地物类型。",
    "这张卫星影像拍摄的是什么场景？",
]
Q_BINARY = [
    "这张遥感影像中是否包含{x}？",
    "影像里能看到{x}吗？",
]
Q_MULTI = [
    "请列出这张遥感影像中出现的地物类别。",
    "这张影像里可能包含哪些地物？",
]
Q_DESC = [
    "请描述这张遥感影像的内容。",
    "描述一下这张卫星图片。",
]


def make_examples(rec, rng):
    """rec 为打标记录；返回该图的指令列表（答案基于 CLIP 标签）。"""
    cls = rec["top1"]
    zh = CLASS_ZH[cls][0]
    exs = []

    # 1) 分类
    exs.append({
        "conversations": [
            {"from": "human", "value": "<image>" + rng.choice(Q_CLASS)},
            {"from": "gpt", "value": f"影像中的主要地物是{zh}。"},
        ],
        "images": [rec["path"]],
    })

    # 2) 二元判断：一半问正类，一半问负类
    if rng.random() < 0.6:
        pos = rng.random() < 0.5
        if pos:
            q_cls, ans = cls, f"是的，影像中包含{zh}。"
        else:
            neg = rng.choice([c for c in CLASS_ZH if c not in [t for t, _ in rec["top5"]]])
            q_cls, ans = neg, f"没有，这张影像中不包含{CLASS_ZH[neg][0]}，主要地物是{zh}。"
        x = CLASS_ZH[q_cls][0]
        exs.append({
            "conversations": [
                {"from": "human", "value": "<image>" + rng.choice(Q_BINARY).format(x=x)},
                {"from": "gpt", "value": ans},
            ],
            "images": [rec["path"]],
        })

    # 3) 描述（模板图注）
    if rng.random() < 0.5:
        desc = f"这张遥感影像展示了一片{zh}区域，{CLASS_ZH[cls][1]}。"
        top2 = rec["top5"][1]
        if top2[1] > 0.10:
            desc += f"画面中还能辨认出{CLASS_ZH[top2[0]][0]}的特征。"
        exs.append({
            "conversations": [
                {"from": "human", "value": "<image>" + rng.choice(Q_DESC)},
                {"from": "gpt", "value": desc},
            ],
            "images": [rec["path"]],
        })

    # 4) 多标签列举
    if rng.random() < 0.3:
        cands = [c for c, p in rec["top5"][:3] if p >= 0.02]
        names = "、".join(CLASS_ZH[c][0] for c in cands)
        exs.append({
            "conversations": [
                {"from": "human", "value": "<image>" + rng.choice(Q_MULTI)},
                {"from": "gpt", "value": f"影像中可能包含：{names}。其中最主要的是{zh}。"},
            ],
            "images": [rec["path"]],
        })
    return exs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--out-clean", required=True, type=Path)
    ap.add_argument("--out-raw", required=True, type=Path)
    ap.add_argument("--out-raw-matched", required=True, type=Path,
                    help="等量对照组：从 raw 中随机抽取与 clean 相同数量的图片")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    labels = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
    keep = {d["path"]: d["keep"] for d in (json.loads(l) for l in open(args.decisions, encoding="utf-8") if l.strip())}

    # 只用 train/validation 做训练数据，test 留给评测
    train_recs = [r for r in labels if "/test/" not in r["path"]]
    clean_recs = [r for r in train_recs if keep.get(r["path"], False)]

    raw_out, clean_out, matched_out = [], [], []
    for rec in train_recs:
        raw_out += make_examples(rec, rng)
    for rec in clean_recs:
        clean_out += make_examples(rec, rng)

    # 等量对照：数据量与 clean 一致、不做置信度过滤（隔离"质量"与"数量"两个变量）
    rng_match = random.Random(args.seed + 1)
    matched_recs = rng_match.sample(train_recs, len(clean_recs)) if clean_recs else []
    for rec in matched_recs:
        matched_out += make_examples(rec, rng_match)

    for path, data in [
        (args.out_clean, clean_out),
        (args.out_raw, raw_out),
        (args.out_raw_matched, matched_out),
    ]:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    n_img_raw, n_img_clean = len(train_recs), len(clean_recs)
    print(f"train images: raw={n_img_raw} clean={n_img_clean} ({n_img_clean / n_img_raw:.1%})")
    print(f"instructions: raw={len(raw_out)} clean={len(clean_out)} raw_matched={len(matched_out)}")
    print(f"-> {args.out_raw}\n-> {args.out_clean}\n-> {args.out_raw_matched}")


if __name__ == "__main__":
    main()
