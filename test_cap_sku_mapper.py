

import sys, logging
sys.path.insert(0, "/home/claude")
logging.basicConfig(level=logging.WARNING)   # suppress debug noise during tests

from cap_sku_mapper import (
    extract_brand, extract_package_type,
    map_caps_to_skus,
    cap_tuples_to_dicts, sku_tuples_to_dicts,
    MAX_MATCH_DIST_P2, MAX_MATCH_DIST_P3,
)

# ── Class dictionaries ────────────────────────────────────────────────────────
CAP_CLASSES = {
    0: "Coke Black Cap", 1: "Coke Glass Caps", 2: "Coke Red Cap",
    3: "Fanta Blue Cap",  4: "Fanta Glass Cap", 5: "Fanta Green Cap",
    6: "Fanta Orange Cap",7: "Fanta Red Cap",   8: "Kinley Bottle Cap",
    9: "Kinley Glass Cap",10:"Kinley Soda Cap", 11:"Other  Caps",
    12:"Other Caps",      13:"Other Glass Cap", 14:"Other Water Caps",
    15:"Sprite Black Cap",16:"Sprite Blue Cap", 17:"Sprite Glass Cap",
    18:"Sprite Green Cap",19:"others",
}
SKU_CLASSES = {
    0: "Coca-Cola 1000ml PET",  1: "Coca-Cola 1500ml PET",
    2: "Coca-Cola 2000ml PET",  3: "Coca-Cola 2250ml PET",
    4: "Coca-Cola 250ml Glass", 5: "Coca-Cola 250ml PET",
    6: "Coca-Cola 500ml PET",   7: "Fanta Lemon 1000ml",
    8: "Fanta Lemon 2000ml PET",9: "Fanta Lemon 2250ml PET",
    10:"Fanta Lemon 250ml",    11:"Fanta Orange 1000ml PET",
    12:"Fanta Orange 1500ml PET",13:"Fanta Orange 2000ml PET",
    14:"Fanta Orange 2250ml PET",15:"Fanta Orange 250ml Glass",
    16:"Fanta Orange 250ml PET",17:"Fanta Orange 500ml PET",
    18:"Kinley Soda 250ml Glass",19:"Kinley Soda 250ml PET",
    20:"Kinley Soda 500ml PET", 21:"Kinley Water 1000ml PET",
    22:"Kinley Water 500ml PET",23:"Other CAN",
    24:"Other PET",             25:"Other RGB",
    26:"Other TPK",             27:"Other Water Bottle",
    28:"Shelf-detection",       29:"Sprite 1000ml PET",
    30:"Sprite 1500ml PET",     31:"Sprite 2000ml PET",
    32:"Sprite 2250ml PET",     33:"Sprite 250ml Glass",
    34:"Sprite 250ml PET",      35:"Sprite 500ml PET",
}

OK   = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
total_fail = 0


def check(label, got, want):
    global total_fail
    ok = got == want
    if not ok:
        total_fail += 1
    print(f"  {OK if ok else FAIL}  {label}")
    if not ok:
        print(f"       got  = {got!r}")
        print(f"       want = {want!r}")
    return ok


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cap(cap_class_id, x1, x2, y1=100, y2=150,
         store="S1", img="img.jpg", iid=1, shelf=1, conf=None):
    d = {"store_id": store, "image_file_name": img, "iteration_id": iid,
         "cap_class_id": cap_class_id,
         "x1": x1, "x2": x2, "y1": y1, "y2": y2,
         "shelfnumber": shelf, "s3path_annotated_file": f"MR/{img}",
         "prod_class_id": None}
    if conf is not None:
        d["confidence"] = conf
    return d

def _sku(prod_class_id, x1, x2, y1=200, y2=500,
         store="S1", img="img.jpg", iid=1, shelf=1, conf=None):
    d = {"store_id": store, "image_file_name": img, "iteration_id": iid,
         "prod_class_id": prod_class_id,
         "x1": x1, "x2": x2, "y1": y1, "y2": y2,
         "shelfnumber": shelf, "brand_name": "", "s3path_annotated_file": f"MR/{img}"}
    if conf is not None:
        d["confidence"] = conf
    return d

def run(caps, skus):
    """Deep-copy inputs and run mapper (mapper mutates dicts in place)."""
    import copy
    return map_caps_to_skus(copy.deepcopy(caps), copy.deepcopy(skus),
                             CAP_CLASSES, SKU_CLASSES)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 1. Brand extraction ══════════════════════════════════════════════════")
cases = [
    ("Coke Red Cap",          "coca-cola"),
    ("Coke Glass Caps",       "coca-cola"),
    ("Sprite Blue Cap",       "sprite"),
    ("Fanta Orange Cap",      "fanta"),
    ("Kinley Glass Cap",      "kinley"),
    ("Other Caps",            ""),
    ("others",                ""),
    ("Coca-Cola 250ml Glass", "coca-cola"),
    ("Sprite 500ml PET",      "sprite"),
]
for name, want in cases:
    check(f"brand('{name}')", extract_brand(name), want)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 2. Package-type extraction ════════════════════════════════════════════")
cases = [
    ("Coke Glass Caps",       "glass"),
    ("Kinley Glass Cap",      "glass"),
    ("Coke Red Cap",          "pet"),
    ("Sprite Blue Cap",       "pet"),
    ("Coke Black Cap",        "pet"),
    ("Kinley Bottle Cap",     "pet"),
    ("Coca-Cola 250ml Glass", "glass"),
    ("Sprite 500ml PET",      "pet"),
    ("Other Caps",            ""),
]
for name, want in cases:
    check(f"pkg('{name}')", extract_package_type(name), want)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 3. Basic P1 vertical mapping ══════════════════════════════════════════")
skus = [
    _sku(4,  0,   100),   # Coca-Cola 250ml Glass
    _sku(5,  100, 200),   # Coca-Cola 250ml PET
    _sku(33, 200, 300),   # Sprite 250ml Glass
    _sku(35, 300, 400),   # Sprite 500ml PET
    _sku(16, 400, 500),   # Fanta Orange 250ml PET
    _sku(18, 500, 600),   # Kinley Soda 250ml Glass
]
caps = [
    _cap(1,  20,  80),   # Coke Glass → Coca-Cola 250ml Glass
    _cap(2,  120, 180),  # Coke Red   → Coca-Cola 250ml PET
    _cap(17, 220, 280),  # Sprite Glass → Sprite 250ml Glass
    _cap(16, 320, 380),  # Sprite Blue  → Sprite 500ml PET
    _cap(6,  420, 480),  # Fanta Orange → Fanta Orange 250ml PET
    _cap(9,  520, 580),  # Kinley Glass → Kinley Soda 250ml Glass
]
expected = [4, 5, 33, 35, 16, 18]
res = run(caps, skus)
for i, (r, e) in enumerate(zip(res, expected)):
    check(f"  cap[{i}] {CAP_CLASSES[caps[i]['cap_class_id']]} → {SKU_CLASSES[e]}",
          r["prod_class_id"], e)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 4. Cross-brand isolation ══════════════════════════════════════════════")
# Sprite cap must never land on Coke SKU even if closer
skus = [_sku(5, 0, 200), _sku(34, 250, 350)]   # wide Coke, narrow Sprite
caps = [_cap(16, 280, 320)]                      # Sprite Blue → Sprite 250ml PET
res = run(caps, skus)
check("Sprite Blue Cap near wide Coke → Sprite 250ml PET", res[0]["prod_class_id"], 34)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 5. Glass vs PET preference ════════════════════════════════════════════")
# Glass cap must prefer Glass SKU even if PET is closer
skus = [_sku(5, 0, 200), _sku(4, 300, 500)]  # PET close, Glass far
caps = [_cap(1, 320, 480)]                     # Coke Glass → must prefer Glass
res = run(caps, skus)
check("Coke Glass Cap prefers Glass SKU over closer PET", res[0]["prod_class_id"], 4)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 6. P4 synthetic fallback (no brand match) ═════════════════════════════")
skus = [_sku(34, 0, 200)]   # Sprite only
caps = [_cap(4,  50, 150)]   # Fanta Glass Cap → no Fanta SKU → P4
res = run(caps, skus)
check("Fanta cap with no Fanta SKU → prod_class_id=-1", res[0]["prod_class_id"], -1)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 7. 'Other Caps' / 'others' always go to P4 ════════════════════════════")
skus = [_sku(5, 0, 500), _sku(34, 0, 500), _sku(16, 0, 500)]  # lots of SKUs
for cap_id, name in [(11, "Other  Caps"), (12, "Other Caps"), (19, "others")]:
    caps = [_cap(cap_id, 50, 150)]
    res = run(caps, skus)
    check(f"'{name}' → P4 (prod_class_id=-1)", res[0]["prod_class_id"], -1)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 8. Duplicate cap deduplication (IoU-based) ════════════════════════════")
# Two nearly identical Coke Red Caps; only one should survive
caps = [
    _cap(2, 100, 200, conf=0.9),   # high conf
    _cap(2, 105, 205, conf=0.4),   # near-duplicate, lower conf
]
skus = [_sku(5, 50, 250)]
res = run(caps, skus)
check("Two overlapping Coke Red Caps deduped to 1", len(res), 1)
check("Surviving cap is the high-conf one (mapped to Coca-Cola 250ml PET)",
      res[0]["prod_class_id"], 5)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 9. Duplicate SKU deduplication (IoU-based) ════════════════════════════")
# Same Sprite SKU detected twice; both caps should still get one each
caps = [_cap(16, 20, 80), _cap(16, 120, 180)]           # 2 Sprite Blue caps
skus = [
    _sku(34, 0,   200, conf=0.95),   # Sprite 250ml PET (high conf)
    _sku(34, 10,  190, conf=0.50),   # near-duplicate of above → deduped
    _sku(35, 300, 500),              # Sprite 500ml PET (different box → kept)
]
res = run(caps, skus)
# After dedup: 2 unique SKUs remain (34 and 35)
# cap[0] → 34 (closest), cap[1] → 35 (slot 34 taken)
check("cap[0] → Sprite 250ml PET (slot 0)", res[0]["prod_class_id"], 34)
check("cap[1] → Sprite 500ml PET (slot 1, after 34 occupied)",
      res[1]["prod_class_id"], 35)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 10. One-slot-per-SKU: 3 caps, 2 SKUs ══════════════════════════════════")
# Three Coke Red caps, only two Coke SKUs → third cap must go to P4
caps = [_cap(2, 50,  100), _cap(2, 150, 200), _cap(2, 500, 600)]
skus = [_sku(5, 0,   200), _sku(6,  300, 500)]   # PET250 and PET500
res = run(caps, skus)
check("cap[0] → slot 1 (Coca-Cola 250ml PET)", res[0]["prod_class_id"] in (5, 6), True)
check("cap[1] → slot 2 (Coca-Cola 500ml PET)", res[1]["prod_class_id"] in (5, 6), True)
check("cap[0] and cap[1] got different SKUs", res[0]["prod_class_id"] != res[1]["prod_class_id"], True)
check("cap[2] → P4 (no free Coke slot)", res[2]["prod_class_id"], -1)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 11. Distance threshold P2 ════════════════════════════════════════════")
far = int(MAX_MATCH_DIST_P2) + 200
caps = [_cap(2,  0,   50)]     # Coke Red Cap at x=25
skus = [_sku(5,  far, far+50)] # Coke SKU very far away
res = run(caps, skus)
# No vertical overlap (P1 fails); same brand+pkg but beyond P2 distance
# → P3 also has same brand, but even farther (same SKU); P3 limit is tighter
check(f"Coke cap + Coke SKU {far}px away → P4 (both thresholds exceeded)",
      res[0]["prod_class_id"], -1)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 12. P3 fires when pkg differs but brand matches ════════════════════════")
# No Glass Coke SKU in image → falls to P3 → picks PET Coke
skus = [_sku(5, 50, 150)]    # Coca-Cola 250ml PET only (no Glass)
caps = [_cap(1, 60, 140)]    # Coke Glass Caps → P3 (brand match, pkg mismatch)
res = run(caps, skus)
check("Coke Glass Cap → P3 → Coca-Cola 250ml PET (no glass SKU available)",
      res[0]["prod_class_id"], 5)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 13. Front-facing: cap + body of same bottle ════════════════════════════")
# Cap bbox is fully inside SKU bbox — both should be detected; cap maps to that SKU
caps = [_cap(2, 80, 120, y1=180, y2=220)]   # Coke Red Cap inside bottle top
skus = [_sku(5, 50, 150, y1=150, y2=500)]   # Coca-Cola 250ml PET (larger box)
res = run(caps, skus)
check("Cap bbox inside SKU bbox → correctly maps to that SKU",
      res[0]["prod_class_id"], 5)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 14. Empty caps list ════════════════════════════════════════════════════")
res = run([], [_sku(5, 0, 100)])
check("Empty cap list → empty result", res, [])

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 15. Empty SKUs list ════════════════════════════════════════════════════")
caps = [_cap(2, 0, 100)]
res = run(caps, [])
check("No SKUs → cap goes to P4", res[0]["prod_class_id"], -1)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══ 16. Tuple conversion round-trip ════════════════════════════════════════")
cap_t = ("S1", "img.jpg", 1, 2, 10, 90, 100, 140, None, 1, "MR/img.jpg")
sku_t = ("S1", "img.jpg", 1, 5, 10, 90, 200, 500, 1, "coca-cola", "MR/img.jpg")
cd = cap_tuples_to_dicts([cap_t])[0]
sd = sku_tuples_to_dicts([sku_t])[0]
check("cap tuple → dict: cap_class_id=2", cd["cap_class_id"], 2)
check("cap tuple → dict: x1=10",          cd["x1"], 10)
check("sku tuple → dict: prod_class_id=5", sd["prod_class_id"], 5)
check("sku tuple → dict: brand_name='coca-cola'", sd["brand_name"], "coca-cola")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
if total_fail == 0:
    print(f"\033[92m  ALL TESTS PASSED\033[0m  (total_fail={total_fail})")
else:
    print(f"\033[91m  {total_fail} TEST(S) FAILED\033[0m")
print("=" * 70)
sys.exit(total_fail)
