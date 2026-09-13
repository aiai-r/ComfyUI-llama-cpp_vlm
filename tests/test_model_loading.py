import ast
import inspect
import ntpath
import posixpath
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


class ModelLoadingTests(unittest.TestCase):
    def load_model(self, path_module=posixpath, **overrides):
        # Execute the actual loader without importing ComfyUI or allocating model weights.
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        storage = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LLAMA_CPP_STORAGE")
        method = next(node for node in storage.body if isinstance(node, ast.FunctionDef) and node.name == "load_model")
        method.decorator_list = []
        factory = Mock()
        handler = Mock()
        namespace = {
            "os": SimpleNamespace(path=path_module),
            "folder_paths": SimpleNamespace(models_dir="models"),
            "Llama": factory,
            "Gemma4ChatHandler": handler,
            "Llava15ChatHandler": handler,
            "_MTMD": False,
            "inspect": inspect,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "nodes.py", "exec"), namespace)
        config = {
            "model": "GGUF/model.gguf", "mmproj": "GGUF/mmproj.gguf",
            "chat_handler": "Gemma4", "n_ctx": 8192, "vram_limit": -1,
            "image_min_tokens": 0, "image_max_tokens": 0, "load_mtp": False,
        }
        config.update(overrides)
        state = SimpleNamespace(clean=Mock(), chat_handler=None)
        namespace["load_model"](state, config)
        return factory.call_args.kwargs

    def test_image_decode_over_512_tokens_fits_on_windows_and_linux(self):
        for path_module in (ntpath, posixpath):
            with self.subTest(platform=path_module.__name__):
                kwargs = self.load_model(path_module)
                # Reproduce llama.cpp's non-causal decode invariant for an image batch.
                effective_ubatch = min(kwargs["n_ctx"], kwargs["n_batch"], kwargs["n_ubatch"])
                for image_tokens in (784, 1120, 2048):
                    self.assertGreaterEqual(effective_ubatch, image_tokens)
                self.assertEqual(kwargs["n_batch"], kwargs["n_ubatch"])
                self.assertEqual(kwargs["n_gpu_layers"], -1)
                self.assertEqual(kwargs["model_path"], path_module.join("models", "LLM", "GGUF/model.gguf"))

    def test_image_budget_is_not_reduced_to_default_batch(self):
        for field in ("image_min_tokens", "image_max_tokens"):
            with self.subTest(field=field):
                kwargs = self.load_model(**{field: 4096})
                self.assertEqual(kwargs["n_batch"], 4096)
                self.assertEqual(kwargs["n_ubatch"], 4096)

    def test_small_context_bounds_allocation(self):
        kwargs = self.load_model(n_ctx=1024)
        self.assertEqual(kwargs["n_batch"], 1024)
        self.assertEqual(kwargs["n_ubatch"], 1024)

    def test_large_context_does_not_allocate_context_sized_microbatch(self):
        kwargs = self.load_model(n_ctx=131072)
        self.assertEqual(kwargs["n_batch"], 2048)
        self.assertEqual(kwargs["n_ubatch"], 2048)

    def test_text_and_other_handlers_keep_library_batch_defaults(self):
        for overrides in ({"mmproj": "None"}, {"mmproj": ""}, {"chat_handler": "LLaVA-1.5"}):
            with self.subTest(overrides=overrides):
                kwargs = self.load_model(**overrides)
                self.assertNotIn("n_batch", kwargs)
                self.assertNotIn("n_ubatch", kwargs)


if __name__ == "__main__":
    unittest.main()
