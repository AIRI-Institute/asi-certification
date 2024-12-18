# Certification of Speaker Recognition Models to Additive Perturbations

This code is the official implementation of [Certification of Speaker Recognition Models to Additive Perturbations](https://arxiv.org/abs/2404.18791) article accepted at AAAI-25.

## Key Files:
- `smooth.py`: Contains the core algorithms for smooth model evaluation.
- `our_utils.py`: Includes utility functions essential for the certification process.
- `parser_certify.py`: Parser script for certification.

## Example Usage:

Setting up the enviroment:
```
conda create -n Certification python=3.11.8
conda activate Certification
pip install -r requirements
```

### Downloading datasets

To download the VoxCeleb1 and VoxCeleb2 datasets, please refer to the official SpeechBrain guide available at: SpeechBrain VoxCeleb Speaker Recognition Guide (https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb/SpeakerRec).

Once you have followed the guide and downloaded the datasets, you can use the `convert.py` script provided by SpeechBrain. This script is used to convert audio files to the WAV format.

### Usage

The usage examples for these modules can be found in `main_nootebook.ipynb`. 
Parameters that can be varied include:
- `num_support_val`: Specifies the number of samples for each speaker used to build the speaker centroid.
- `classes_per_it_val`: Determines the number of enrolled speakers that the smooth model can recognize in the system.
- `sigma`: Represents the noise level added to audio samples for evaluating `g(x)`.
- `N`: Denotes the number of noised samples utilized for evaluating `g(x)`.
- `K`: Indicates the maximum number of repeats (R) allowed if R*N noised samples in g(x) fail to satisfy the robustness condition.
- `model_name`: Name of the model – options include 'ecapa-tdnn', 'pyannote', 'wespeaker', 'campplus', 'eres2net', 'wavlm'.
- `emb_size`: Represents the length of embedding vectors for the selected model.

Functions for Plotting Graphs:
- `ERA_of_f`: Computes the Empirical Robustness Accuracy (ERA) for the model.
- `ERA_of_g`: Computes the Empirical Robustness Accuracy (ERA) for the smoothed model.
- `predict_with_radius`: Classifies the speaker and provides the radius within which the input sample x can be perturbed without altering the model prediction.
- `predict_with_radius_2`: Similar to predict_with_radius, this function also classifies the speaker while considering the perturbation radius for the input sample x.

Example of plotted graphs can be found in `Graphs` folder.



## Acknowledgements

Our code is partially based on SE code https://github.com/koava36/certrob-fewshot.

## Citation

If you find this repository and our work useful, please consider giving a star and please cite as:

```bash
@article{korzh2024certification,
  title={Certification of Speaker Recognition Models to Additive Perturbations},
  author={Korzh, Dmitrii and Karimov, Elvir and Pautov, Mikhail and Rogov, Oleg Y and Oseledets, Ivan},
  journal={arXiv preprint arXiv:2404.18791},
  year={2024}
}
```
