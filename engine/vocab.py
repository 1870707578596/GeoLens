"""共享类别词表与中文回答解析。

数据来源：scripts/20_build_instructions.py 的中文知识库与 eval/30_zeroshot_eval.py
的同义词表，v2 起统一收敛到本模块（旧脚本后续逐步改为引用此处）。
"""

import re

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

ZH2EN = {v[0]: k for k, v in CLASS_ZH.items()}

# 模型回答里的常见说法 -> 标准类名；有歧义的（如"农田""住宅区"）不映射
SYNONYMS = {
    "飞机场": "机场", "高速路": "高速公路", "铁轨": "铁路", "码头": "港口",
    "桥": "桥梁", "山地": "山脉", "云层": "云", "灌木": "灌木丛", "沙滩": "海滩",
    "轮船": "船舶", "船只": "船舶", "储罐": "储油罐", "油罐": "储油罐",
    "发电厂": "火力发电站", "电厂": "火力发电站", "跑道": "飞机跑道",
}

_ALL_NAMES = dict(ZH2EN)
_ALL_NAMES.update({k: ZH2EN[v] for k, v in SYNONYMS.items()})  # 同义词 -> 英文类名
_ALL_PATTERN = "|".join(sorted(_ALL_NAMES.keys(), key=len, reverse=True))


def parse_class(answer: str) -> str | None:
    """从模型的中文回答里解析类别（英文类名）；解析失败返回 None。

    标准名与同义词合并按长度降序匹配，保证"飞机场"不会被子串"飞机"截胡。
    """
    if not answer:
        return None
    hits = re.findall(_ALL_PATTERN, answer)
    return _ALL_NAMES[hits[0]] if hits else None
