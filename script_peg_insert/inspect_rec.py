import torch

# Load the trajectories file
data = torch.load("script_peg_insert/state_records.pt")

print("=" * 80)
print("State Records Content Inspection")
print("=" * 80)

# Check the type of the loaded data
print(f"\nData type: {type(data)}")

# If it's a dictionary, print keys and their contents
if isinstance(data, dict):
    print(f"\nNumber of keys: {len(data.keys())}")
    print("\nKeys in the file:")
    for key in data.keys():
        print(f"  - {key}")
    
    print("\n" + "=" * 80)
    print("Detailed Content:")
    print("=" * 80)
    
    for key, value in data.items():
        print(f"\n[{key}]")
        print(f"  Type: {type(value)}")
        
        if isinstance(value, torch.Tensor):
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Device: {value.device}")
            print(f"  First few elements: {value.flatten()[:10]}")
        elif isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                if isinstance(value[0], torch.Tensor):
                    print(f"  First element shape: {value[0].shape}")
                print(f"  First element: {value[0]}")
        elif isinstance(value, dict):
            print(f"  Sub-keys: {list(value.keys())}")
            for sub_key, sub_value in value.items():
                print(f"    [{sub_key}]")
                print(f"      Type: {type(sub_value)}")
                if isinstance(sub_value, torch.Tensor):
                    print(f"      Shape: {sub_value.shape}")
                    print(f"      Dtype: {sub_value.dtype}")
        else:
            print(f"  Value: {value}")

elif isinstance(data, (list, tuple)):
    print(f"\nLength: {len(data)}")
    print(f"Element type: {type(data[0]) if len(data) > 0 else 'N/A'}")
    if len(data) > 1 and isinstance(data[1], dict):
        entry = data[1]
        # Case 1: list of per-step dicts with {"step": int, "state": {...}}
        if "state" in entry and "step" in entry:
            print("\nDetected step-wise records; showing items 1-3:")
            for i, item in enumerate(data[1:4], start=1):
                print(f"\nItem index {i} (step={item.get('step')}):")
                state = item.get("state", {})
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, first elems={v.flatten()[:6]}")
                    else:
                        print(f"  {k}: {v}")
        # Case 2: trajectory dict where each key maps to a list of tensors
        else:
            keys = list(entry.keys())
            print(f"Second trajectory keys: {keys}")
            print("\nSecond trajectory - first 3 steps:")
            # determine how many steps are present (use first iterable value length)
            iterable_values = [v for v in entry.values() if isinstance(v, (list, tuple))]
            max_steps = min(len(v) for v in iterable_values) if iterable_values else 0
            for step in range(min(3, max_steps)):
                print(f"\nStep {step}:")
                for k in keys:
                    vals = entry.get(k, [])
                    if isinstance(vals, (list, tuple)) and step < len(vals):
                        v = vals[step]
                        if isinstance(v, torch.Tensor):
                            print(
                                f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, first elems={v.flatten()[:6]}"
                            )
                        else:
                            print(f"  {k}: {v}")
                    else:
                        print(f"  {k}: (no data)")
    else:
        print("\nFirst few elements:")
        for i, item in enumerate(data[:3]):
            print(f"  [{i}]: {item}")

elif isinstance(data, torch.Tensor):
    print(f"\nShape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Device: {data.device}")
    print(f"\nFirst few elements:\n{data.flatten()[:20]}")
    print(f"\nFull tensor (if small):\n{data if data.numel() < 100 else '(too large to display)'}")

else:
    print(f"\nContent:\n{data}")

print("\n" + "=" * 80)
