"""Pipeline test tool - batch mode and interactive mode."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.inference.pipeline import create_pipeline
from ml.data.data_loader import VenueData, regenerate_route, parse_route_id


def test_route(pipeline, venue, route_id):
    parsed = parse_route_id(route_id)
    route_data = regenerate_route(venue, parsed)
    if not route_data:
        print(f"  SKIP: could not regenerate {route_id}")
        return

    steps = route_data.get('steps', [])
    is_mf = route_data.get('is_multi_floor', False)
    tag = " [MULTI-FLOOR]" if is_mf else ""
    print(f"\n{'='*70}")
    print(f"  ROUTE: {route_id}{tag}")
    print(f"  Steps: {len(steps)}")
    print(f"{'='*70}")

    result = pipeline.predict(route_data)

    print("\n--- Step Filter ---")
    for d in result['all_step_decisions']:
        status = 'KEEP' if d['keep'] else 'DEL '
        print(f"  Step {d['step_number']:2d} ({d['action']:12s}) -> {status}  (p={d['keep_prob']:.3f})")

    print("\n--- Description Strategy ---")
    for d in result['strategy_decisions']:
        strat = 'ANCHOR    ' if d['anchor_based'] else 'NON-ANCHOR'
        print(f"  Step {d['step_number']:2d} -> {strat}  (p={d['anchor_prob']:.3f})")

    print("\n--- Anchor Selection ---")
    for d in result['anchor_decisions']:
        sel = d['selected_anchor']
        if isinstance(sel, (list, tuple)) and len(sel) >= 2:
            name = f"{sel[0]}-{sel[1]}"
        else:
            name = str(sel)
        probs = ', '.join(f"{p:.2f}" for p in d['proba'])
        print(f"  Step {d['step_number']:2d} -> {name}  [{probs}]")

    print("\n--- Generated Descriptions ---")
    for d in result['generated_descriptions']:
        print(f"  Step {d['step_number']:2d} [{d['source']:8s}]: {d['description']}")

    print("\n--- Final Route Description ---")
    for d in result['generated_descriptions']:
        print(f"  {d['description']}")


def list_rooms(venue):
    """List all available rooms grouped by floor."""
    print(f"\n{'='*70}")
    print("  AVAILABLE ROOMS")
    print(f"{'='*70}")
    for i, floor_name in enumerate(venue.floor_names):
        areas = venue.floor_areas[i]
        rooms = []
        for room_type, room_list in areas.items():
            if room_type in ('Building', 'Walking', 'Water', 'Green'):
                continue
            for room in room_list:
                rid = room.get('id', '')
                if rid:
                    rooms.append(f"{room_type} - {rid}")
        rooms.sort()
        print(f"\n  [{floor_name}] ({len(rooms)} rooms)")
        for j in range(0, len(rooms), 4):
            row = rooms[j:j+4]
            print(f"    {', '.join(row)}")


def interactive_mode(pipeline, venue):
    """Interactive test: user picks start/end rooms."""
    list_rooms(venue)

    print(f"\n{'='*70}")
    print("  INTERACTIVE MODE")
    print("  Format:  <floor>/<type>-<id>  (e.g. Kat -1/Shop-ID-153)")
    print("  Type 'list' to show rooms, 'quit' to exit")
    print(f"{'='*70}")

    while True:
        print()
        start_input = input("  Start > ").strip()
        if start_input.lower() in ('quit', 'q', 'exit'):
            break
        if start_input.lower() == 'list':
            list_rooms(venue)
            continue

        end_input = input("  End   > ").strip()
        if end_input.lower() in ('quit', 'q', 'exit'):
            break
        if end_input.lower() == 'list':
            list_rooms(venue)
            continue

        start_floor, start_type, start_id = _parse_room_input(start_input)
        end_floor, end_type, end_id = _parse_room_input(end_input)

        if not start_floor or not end_floor:
            print("  ERROR: Invalid format. Use: Kat -1/Shop-ID-153")
            continue

        route_id = f"{start_floor}_{start_type}_{start_id}_to_{end_floor}_{end_type}_{end_id}"
        print(f"  Route ID: {route_id}")
        test_route(pipeline, venue, route_id)


def _parse_room_input(text: str):
    """Parse 'Kat -1/Shop-ID-153' into (floor, type, id)."""
    text = text.strip()

    if '/' in text:
        floor_part, room_part = text.split('/', 1)
    else:
        print(f"  ERROR: Missing '/' separator in '{text}'")
        return None, None, None

    floor_part = floor_part.strip()

    room_part = room_part.strip()
    if '-' in room_part:
        dash_idx = room_part.index('-')
        room_type = room_part[:dash_idx].strip()
        room_id = room_part[dash_idx+1:].strip()
    else:
        print(f"  ERROR: Could not parse room from '{room_part}'")
        return None, None, None

    return floor_part, room_type, room_id


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-t5', action='store_true', help='Enable T5 text gen model')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode: pick start/end rooms')
    args = parser.parse_args()

    print(f"Loading pipeline (use_text_gen={args.use_t5})...")
    pipeline = create_pipeline(use_text_gen=args.use_t5)

    print("Loading venue data...")
    venue = VenueData(venue='zorlu')
    venue.load()

    if args.interactive:
        interactive_mode(pipeline, venue)
    else:
        test_routes = [
            'Kat -1_Food_ID-123_to_Kat -1_Shop_ID-170',
            'Kat -1_Food_ID-133_to_Kat -1_Food_ID-134',
            'Kat -2_Shop_ID-207_to_Kat -1_Food_ID-118',
            'Kat -2_Shop_ID-237_to_Kat -1_Shop_ID-149',
            'Kat 1_Shop_ID101_to_Kat -2_Shop_ID-226',
        ]

        for rid in test_routes:
            test_route(pipeline, venue, rid)

    print("\n\nDone.")


if __name__ == '__main__':
    main()
