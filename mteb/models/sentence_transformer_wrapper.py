from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb.encoder_interface import PromptType
from mteb.models.wrapper import Wrapper

logger = logging.getLogger(__name__)


class SentenceTransformerWrapper(Wrapper):
    def __init__(
        self,
        model: str | SentenceTransformer | CrossEncoder,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for SentenceTransformer models.

        Args:
            model: The SentenceTransformer model to use. Can be a string (model name), a SentenceTransformer model, or a CrossEncoder model.
            revision: The revision of the model to use.
            model_prompts: A dictionary mapping task names to prompt names.
                First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                and finally to the specific prompt type.
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        if isinstance(model, str):
            self.model = SentenceTransformer(model, revision=revision, **kwargs)
        else:
            self.model = model

        if (
            model_prompts is None
            and hasattr(self.model, "prompts")
            and len(self.model.prompts) > 0
        ):
            try:
                model_prompts = self.validate_task_to_prompt_name(self.model.prompts)
            except KeyError:
                model_prompts = None
                logger.warning(
                    "Model prompts are not in the expected format. Ignoring them."
                )
        elif model_prompts is not None and hasattr(self.model, "prompts"):
            logger.info(f"Model prompts will be overwritten with {model_prompts}")
            self.model.prompts = model_prompts
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

        if isinstance(self.model, CrossEncoder):
            self.predict = self._predict

        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            self.similarity = self.model.similarity

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)


        Returns:
            The encoded sentences.
        """
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(
                self.model_prompts, task_name, prompt_type
            )
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(sentences)} sentences.")

        embeddings = self.model.encode(
            sentences,
            prompt_name=prompt_name,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings
    
    def get_text_embeddings(
            self,
            texts: Sequence[str],
            *,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
        
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(
                self.model_prompts, task_name, prompt_type
            )
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(texts)} sentences.")

        embeddings = self.model.encode(
            texts,
            prompt_name=prompt_name,
            **kwargs,
        )
        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings

    
    def get_image_embeddings(
            self,
            images: list[Image.Image] | DataLoader,
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,):
            """Encodes the given sentences using the encoder from sentence-transformers/clip-ViT-L-14.

            """
            all_image_embeddings = []

            if isinstance(images, DataLoader):
                with torch.no_grad():
                    for batch in tqdm(images):
                        # print("image: ", batch)
                        inputs = [to_pil_image(k) for k in batch]
                        image_outputs = self.model.encode(inputs)
                        if isinstance(image_outputs, np.ndarray):
                            all_image_embeddings.append(image_outputs)
                        else: 
                            all_image_embeddings.append(image_outputs.cpu())
            else:
                with torch.no_grad():
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        batch_images = [
                            img.convert("RGB")
                            if isinstance(img, Image.Image) and img.mode != "RGB"
                            else img
                            for img in batch_images
                        ]
                        image_outputs = self.model.encode(batch_images)
                        if isinstance(image_outputs, np.ndarray):
                            all_image_embeddings.append(image_outputs)
                        else: 
                            all_image_embeddings.append(image_outputs.cpu())
                        

            if isinstance(all_image_embeddings[0], np.ndarray):
                all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
            else:
                all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            
            return all_image_embeddings
    
    def calculate_probs(self, text_embeddings, image_embeddings):
        # normalized features
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)

        # convert to torch
        if isinstance(image_embeddings, np.ndarray):
            image_embeddings = torch.from_numpy(image_embeddings)
        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.from_numpy(text_embeddings)

        clip_model = self.model._first_module().model
        logit_scale = getattr(clip_model, "logit_scale", None)
        logit_bias = getattr(clip_model, "logit_bias", None)
        if logit_scale is not None:
            logit_scale = logit_scale.exp()  # usually stored in log space
        if logit_bias is None:
            logit_bias = torch.tensor([0])

        # cosine similarity as logits
        logits_per_text = torch.matmul(
            text_embeddings, image_embeddings.t().to(text_embeddings.device)
        ) * logit_scale.to(text_embeddings.device) + logit_bias.to(text_embeddings.device)
        logits_per_image = logits_per_text.t()
        return logits_per_image

    def _predict(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        return self.model.predict(
            sentences,
            convert_to_numpy=True,
            **kwargs,
        )
