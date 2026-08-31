"""
@author: (community add-on)
@title: LM Studio Multi-Image (Expo)
@description: Adds a node that sends TWO OR MORE images to LM Studio's vision
model in a single chat message, so the model can compare/reason over them
together (instead of the stock I2T / Unified nodes, which only ever look at
the first frame of a single IMAGE batch).

"""

import os
import tempfile
import hashlib
import random
import concurrent.futures

import numpy as np
from PIL import Image

# Reuse everything already defined in the main module instead of duplicating it.
from .expo_lmstudio_imagetotext import (
    lms,
    DEFAULT_VISION,
    check_lmstudio_connection,
    get_model_info_with_fallback,
    _collect_response,
    safe_get_stats_info,
)


def _tensor_batch_to_pil_list(image_tensor):
    """
    ComfyUI IMAGE tensors are batches: shape (N, H, W, C), float 0..1.
    Returns a list of PIL Images, one per item in the batch, so that a
    single IMAGE input carrying multiple frames also gets expanded.
    """
    if image_tensor is None:
        return []
    arr = np.uint8(np.array(image_tensor) * 255)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return [Image.fromarray(frame) for frame in arr]


class ExpoLmstudioImageToTextMulti:
    """
    Same idea as ExpoLmstudioImageToText, but accepts up to 4 image inputs
    (image_1 required, image_2/3/4 optional) and sends ALL of them to the
    model as one multi-image chat message. Any input that is itself a batch
    (N>1) is expanded too, so you can mix single images and batches freely.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "Compare these images and describe the differences."}),
                "system_prompt": ("STRING", {"default": "This is a chat between a user and an assistant. The assistant is an expert in analyzing and comparing images, with detail and accuracy."}),
                "model_key": ("STRING", {"default": DEFAULT_VISION}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 1}),
                "strip_thinking": ("BOOLEAN", {"default": True, "tooltip": "Strip <think>...</think> reasoning blocks from the response (for models with thinking mode enabled)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)
    FUNCTION = "process_images"
    CATEGORY = "ComfyExpo/I2T"

    @classmethod
    def IS_CHANGED(cls, image_1, user_prompt, system_prompt, model_key, auto_unload, unload_delay, seed,
                    image_2=None, image_3=None, image_4=None,
                    max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, strip_thinking=True):
        m = hashlib.sha256()
        for val in (user_prompt, system_prompt, model_key, auto_unload, unload_delay, seed,
                    max_tokens, temperature, debug, timeout_seconds, strip_thinking):
            m.update(str(val).encode())
        for img in (image_1, image_2, image_3, image_4):
            if img is not None:
                m.update(np.array(img).tobytes())
        return m.hexdigest()

    def process_images(self, image_1, user_prompt, system_prompt, model_key, auto_unload, unload_delay, seed,
                        image_2=None, image_3=None, image_4=None,
                        max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, strip_thinking=True):
        debug = debug if isinstance(debug, bool) else str(debug).lower() == "true"
        strip_thinking = strip_thinking if isinstance(strip_thinking, bool) else str(strip_thinking).lower() == "true"

        check_lmstudio_connection()

        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        # Expand every connected IMAGE input (each may itself be a batch).
        pil_images = []
        for img in (image_1, image_2, image_3, image_4):
            pil_images.extend(_tensor_batch_to_pil_list(img))

        if not pil_images:
            return ("No images provided.",)

        if debug:
            print(f"Debug: Starting process_images (multi) method")
            print(f"Debug: Total images to send: {len(pil_images)}")
            print(f"Debug: User prompt: {user_prompt}")
            print(f"Debug: Requested Model: {model_key}")

        managed_temp_paths = []
        timed_out = [False]

        def do_the_work():
            with lms.Client() as client:
                model_key_to_use = get_model_info_with_fallback(model_key, debug)

                if model_key_to_use:
                    if auto_unload == "True" and unload_delay > 0:
                        model_obj = client.llm.model(model_key_to_use, ttl=unload_delay)
                    else:
                        model_obj = client.llm.model(model_key_to_use)
                else:
                    model_obj = client.llm.model()

                chat = lms.Chat(system_prompt)

                # Save every image to its own temp file and prepare a handle for each.
                image_handles = []
                for idx, pil_image in enumerate(pil_images):
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        tmp_path = tmp.name
                        managed_temp_paths.append(tmp_path)
                        pil_image.save(tmp_path, format='JPEG')
                    if debug:
                        print(f"Debug: Saved image {idx + 1}/{len(pil_images)} to {tmp_path}")
                    image_handles.append(client.files.prepare_image(tmp_path))

                # This is the key line: pass ALL handles in one call so the
                # model sees every image within the same user turn.
                chat.add_user_message(user_prompt, images=image_handles)

                config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "seed": seed
                }

                if debug:
                    print(f"Debug: Sending request to LM Studio with config: {config}")

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_collect_response, model_obj, chat, config, strip_thinking)
                    try:
                        result, output_text = future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        error_message = f"Error: LM Studio model response timed out after {timeout_seconds} seconds."
                        print(error_message)
                        return (error_message,)

                stats_info = safe_get_stats_info(result, debug)
                if debug:
                    print(f"Debug: Tokens generated: {stats_info['predicted_tokens']}, Time to first token: {stats_info['time_to_first_token']}s")

                if auto_unload == "True" and unload_delay == 0:
                    try:
                        if debug:
                            print("Debug: Unloading model immediately.")
                        model_obj.unload()
                    except Exception as unload_err:
                        print(f"Warning: Failed to unload model: {unload_err}")

                return (output_text,)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(do_the_work)
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            timed_out[0] = True
            error_message = f"Error: LM Studio operation timed out after {timeout_seconds} seconds. The connection may be unstable."
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"LM Studio error (Image to Text Multi node): {str(e)}"
            print(error_message)
            raise Exception(error_message) from e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if not timed_out[0]:
                for temp_path in managed_temp_paths:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except Exception as cleanup_err:
                            print(f"Warning: Failed to remove temporary file {temp_path}: {cleanup_err}")


NODE_CLASS_MAPPINGS = {
    "Expo Lmstudio Image To Text Multi": ExpoLmstudioImageToTextMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Expo Lmstudio Image To Text Multi": "LM Studio (Image to Text, Multi-Image)",
}
