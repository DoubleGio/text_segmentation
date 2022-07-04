from abc import abstractmethod
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import BatchEncoding
rng = np.random.default_rng()

class TS_Pipeline:

    def __init__(
        self,
        tokenizer,
        model,
        device: torch.device,
        batch_size: int = 8,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._preprocess_params, self._forward_params = self._sanitize_parameters(**kwargs)

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    @abstractmethod
    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Tuple[Dict[str, torch.TensorType], np.ndarray, Optional[Any]]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionnary of everything necessary for
        `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
        """
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def _forward(self, input_tensors: Dict[str, torch.TensorType], **forward_parameters: Dict) -> torch.TensorType:
        """
        _forward will receive the prepared dictionnary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError("_forward not implemented")
        
    def forward(self, model_inputs, **forward_params):
        with torch.inference_mode():
            model_inputs = model_inputs.to(self.device)
            model_output = self._forward(model_inputs, **forward_params)
            model_output = model_output.to('cpu')
        return model_output

    def get_iterator(self, inputs, preprocess_params, forward_params):
        dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        collate_fn = self.no_collate_fn if self.batch_size == 1 else self.cat_collate_fn(self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True if self.device.type == "cuda" else False)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=self.batch_size)
        return model_iterator

    def run_single(self, inputs:str, preprocess_params, forward_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        return model_outputs

    def __call__(self, inputs: Union[List[str], str], **kwargs: Any) -> Any:
        preprocess_params, forward_params = self._sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}

        if isinstance(inputs, list):
            return self.get_iterator(inputs, preprocess_params, forward_params)
        else:
            return self.run_single(inputs, preprocess_params, forward_params)

    @staticmethod
    @abstractmethod
    def no_collate_fn(item):
        """Collate function for batch_size=1"""
        raise NotImplementedError("no_collate_fn not implemented")

    @staticmethod
    @abstractmethod
    def cat_collate_fn(tokenizer) -> Callable:
        """Returns the cat_collate function."""
        raise NotImplementedError("cat_collate_fn not implemented")


class PipelineDataset(Dataset):

    def __init__(self, paths: List[str], process: Callable, process_params: Dict[str, Any]):
        self.paths = paths
        self.process = process
        self.process_params = process_params

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        item = self.process(self.paths[i], **self.process_params)
        if item is None:
            return self.__getitem__(rng.integers(0, len(self.paths)))
        else:
            return item

class PipelineIterator(IterableDataset):

    def __init__(self, loader, forward: Callable, params: Dict[str, Any], loader_batch_size: int):
        self.loader = loader
        self.forward = forward
        self.params = params
        self.loader_batch_size = loader_batch_size

    def __len__(self):
        return len(self.loader.dataset)
    
    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        item = next(self.iterator)
        return self.forward(item[0]), *item[1:]

