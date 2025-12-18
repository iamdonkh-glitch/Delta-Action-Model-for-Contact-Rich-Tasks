# save as script_peg_insert/test_torch.py and run: uv run python script_peg_insert/test_torch.py
# import torch
# from pathlib import Path
# import tempfile

# def supports_weights_only():
#     data = {"x": torch.tensor([1, 2, 3])}
#     with tempfile.TemporaryDirectory() as td:
#         path = Path(td) / "test.pt"
#         try:
#             torch.save(data, path, weights_only=True)
#             torch.load(path)  # ensure file is readable
#             return True
#         except TypeError as e:
#             print(f"TypeError: {e}")
#             return False
#         except Exception as e:
#             print(f"Other exception: {e}")
#             return False

# if __name__ == "__main__":
#     ok = supports_weights_only()
#     print(f"weights_only supported: {ok}")

# save as script_peg_insert/test_torch_load.py and run: uv run python script_peg_insert/test_torch_load.py
import torch
from pathlib import Path
import tempfile

def load_supports_weights_only():
    obj = {"w": torch.tensor([1.0, 2.0, 3.0])}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tmp.pt"
        torch.save(obj, path)  # plain save
        try:
            torch.load(path, weights_only=True)
            return True
        except TypeError as e:
            print(f"TypeError: {e}")
            return False
        except Exception as e:
            print(f"Other exception: {e}")
            return False

if __name__ == "__main__":
    supported = load_supports_weights_only()
    print(f"torch.load weights_only supported: {supported}")
