import pandas as pd
import argparse
import os

FIXED_10_CLASSES = [
    "tench",  # 0: Tench (fish)
    "English springer",  # 217: English Springer Spaniel
    "cassette player",  # 486: Cassette Player
    "chain saw",  # 491: Chain Saw
    "church",  # 497: Church
    "French horn",  # 566: French Horn
    "garbage truck",  # 569: Garbage Truck
    "gas pump",  # 571: Gas Pump
    "golf ball",  # 574: Golf Ball
    "parachute"  # 701: Parachute
]

SIMILAR_OBJECTS_MAP = {
    "tench": ["fish", "goldfish", "great white shark", "barracuda"],
    "English springer": ["dog", "beagle", "border collie", "golden retriever"],
    "cassette player": ["radio", "boombox", "speaker", "mp3 player"],
    "chain saw": ["handsaw", "drill", "hammer", "power tool"],
    "church": ["house", "barn", "castle", "temple", "building"],
    "French horn": ["trumpet", "saxophone", "trombone", "flute"],
    "garbage truck": ["truck", "bus", "van", "car", "trailer"],
    "gas pump": ["vending machine", "atm", "parking meter", "telephone booth"],
    "golf ball": ["baseball", "tennis ball", "ping pong ball", "white ball"],
    "parachute": ["balloon", "kite", "umbrella", "cloud", "sky"]
}


def main():
    parser = argparse.ArgumentParser(description="Auto-Generate dataset for fixed 10 classes")
    parser.add_argument('--target', type=str, required=True,
                        help="Name of the object to erase (must be in the predefined 10 classes)")
    parser.add_argument('--output', type=str, default=None, help="Output file path (auto-named if not specified)")
    args = parser.parse_args()

    target_clean = args.target.lower().strip()

    matched_target = None
    for c in FIXED_10_CLASSES:
        if target_clean in c.lower():
            matched_target = c
            break

    if not matched_target:
        print(f"❌ Error: The input target '{args.target}' is not in the fixed 10-class list!")
        print(f"Available classes: {FIXED_10_CLASSES}")
        return

    print(f"[INFO] Target identified: {matched_target}")

    preserve_list = []

    for c in FIXED_10_CLASSES:
        if c != matched_target:
            preserve_list.append(c)

    found_similar = False
    for key, sims in SIMILAR_OBJECTS_MAP.items():
        if matched_target.lower() in key.lower():
            preserve_list.extend(sims)
            found_similar = True
            print(f"[INFO] Auto-added similar objects: {sims}")
            break

    if not found_similar:
        pass

    preserve_list = list(set(preserve_list))

    templates = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a dark photo of the {}",
        "a picture of a {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped shot of a {}",
        "the {} in the video game",
        "a sketch of a {}",
        "an embroidered {}",
        "a difficult drawing of a {}",
        "a photo of the dirty {}",
        "a painting of the {}",
        "art of the {}",
        "graffiti of a {}",
        "a drawing of the {}",
        "a plastic {}",
        "a photo of the hard to see {}",
        "a low resolution photo of the {}",
        "a rendering of the {}",
        "a blurry photo of a {}",
        "a black and white photo of the {}",
        "a low contrast photo of the {}",
        "a high contrast photo of the {}",
        "a bad photo of the {}",
        "a good photo of the {}",
        "a photo of the small {}",
        "a photo of the large {}",
        "image of {}",
        "a view of {}",
        "{} in the distance",
        "a {} background",
        "a realistic {}"
    ]

    data = []
    case_id = 0

    for t in templates:
        prompt = t.format(matched_target)
        data.append({
            "case_number": case_id,
            "prompt": prompt,
            "label": "target",
            "class": matched_target
        })
        case_id += 1

    for p_obj in preserve_list:
        for t in templates:
            prompt = t.format(p_obj)
            data.append({
                "case_number": case_id,
                "prompt": prompt,
                "label": "preserve",
                "class": p_obj
            })
            case_id += 1

    df = pd.DataFrame(data)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        safe_name = matched_target.replace(" ", "_")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        output_path = os.path.join(project_root, "data", f"object_prompts_{safe_name}.csv")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_path, index=False)

    print(f"✅ Dataset successfully generated and saved to: {output_path}")


if __name__ == "__main__":
    main()