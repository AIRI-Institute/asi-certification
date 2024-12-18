import os
import torch, torchaudio
import wespeaker
import glob
import math
from typing import List, Tuple, Union, Optional
import random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
from time import time
import datetime
import torch.nn.functional as F
from tqdm import tqdm
import gc

from pyannote.audio import Model
from pyannote.audio import Inference

from speechbrain.inference.speaker import EncoderClassifier

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


def ERA_of_f(model, producer, device, delta_max, n_grid=11, batch_size=64, attack="normal", audio_len=3):
    """
    Calculate empirical robust accuracy of base classifier f for set of audios
    and for max level of noise perturbation delta_max

    """
    class_prototypes, speaker_enrollment_audios, speaker_inference_audios, id2class, class2id = producer.produce_subsets()
    class_prototypes_list = torch.stack(list(class_prototypes.values()))
    centroids = class_prototypes_list.squeeze(1).to(device)
    model.eval()
    model.to(device)
    results = torch.ones(len(producer.speakers_test_only))
    # print(centroids.shape)
    attack_levels = np.linspace(0, delta_max, n_grid)
    results2 = torch.zeros(len(producer.speakers_test_only), n_grid)

    # with torch.no_grad():
    for k, speaker_id in tqdm(enumerate(producer.speakers_test_only), total=len(producer.speakers_test_only)):
        gt_class = id2class[speaker_id]
        wavs_paths = speaker_inference_audios[speaker_id]
        np.random.seed(42)
        wav_path = wavs_paths[np.random.choice(len(wavs_paths))]
        sample = torchaudio.load(wav_path)[0][0, :audio_len * 16000].to(device)

        for j, attack_level in enumerate(attack_levels):
            ## MIGHT BE PDG, FGSM
            if attack == "normal":
                noise = torch.randn_like(sample.repeat(batch_size, 1), device=device)
                noise = noise / torch.norm(noise, dim=-1, keepdim=True) * attack_level
                x = sample.repeat(batch_size, 1)
                qq = x + noise
            elif attack == "pgd":
                qq = []

                sample.requires_grad = True
                assert sample.requires_grad
                assert sample.is_leaf

                for _ in range(batch_size):
                    pert_sample = pgd(model, sample, model(sample), criterion=torch.nn.CosineSimilarity(dim=-1),
                                      eps=0.002, eps_step=0.002, clip_min=-1, clip_max=1, targeted=False, device=device)
                    qq.append(sample + (pert_sample - sample) / torch.norm(pert_sample - sample) * attack_level)
                qq = torch.stack(qq)
            elif attack == "anonymization":
                path_to_anon_noise = # path to UAP
                anon_noise, _ = torchaudio.load(path_to_anon_noise)
                anon_noise = anon_noise.to(device)
                anon_noise = attack_level * anon_noise / anon_noise.norm()
                
                x_len = sample.shape[-1]
                
                noise = anon_noise.repeat(batch_size, math.ceil(x_len / len(anon_noise)))
                noise = noise[:, :x_len]
                
                x = sample.repeat(batch_size, 1)
                qq = x + noise
            
            with torch.no_grad():
                batch_emb = model(qq)
                norms = torch.norm(batch_emb, p=2, dim=-1, keepdim=True)
                batch_emb /= norms
                batch_emb = batch_emb.reshape(-1, 1, model.emb_size)
                # print(batch_emb.shape, centroids.shape)

                cos_sim = F.cosine_similarity(batch_emb, centroids, dim=-1)
                predicted_classes = cos_sim.argmax(dim=1)
                # print(predicted_classes.shape)
                all_correct = torch.all(predicted_classes == gt_class).item()
                results[k] = results[k] * all_correct
                results2[k, j] = all_correct

                if not all_correct:
                    break
    era = torch.mean(results).item()
    return results, results2, era


def ERA_of_g(model, smoothed_model, args, producer, device, delta_max, n_grid=3, batch_size=2, attack="normal", audio_len=3):
    """
    Calculate empirical robust accuracy of SMOOTHED classifier g for set of audios
    and for max level of noise perturbation delta_max

    """
    class_prototypes, speaker_enrollment_audios, speaker_inference_audios, id2class, class2id = producer.produce_subsets()
    class_prototypes_list = torch.stack(list(class_prototypes.values()))
    centroids = class_prototypes_list.squeeze(1).to(device)

    results = torch.ones(len(producer.speakers_test_only))

    attack_levels = np.linspace(0, delta_max, n_grid)
    results2 = torch.zeros(len(producer.speakers_test_only), n_grid)

    for k, speaker_id in tqdm(enumerate(producer.speakers_test_only),
                                        total=len(producer.speakers_test_only)):
        gt_class = id2class[speaker_id]

        wavs_paths = speaker_inference_audios[class2id[gt_class]]
        np.random.seed(42)
        wav_path = wavs_paths[np.random.choice(len(wavs_paths))]
        sample = torchaudio.load(wav_path)[0][0, :audio_len * 16000].to(device)

        for j, attack_level in enumerate(attack_levels):
            ## MIGHT BE PDG, FGSM, but probably theb bs is 1
            if attack == "normal":
                noise = torch.randn_like(sample.repeat(batch_size, 1), device=device)
                noise = noise / torch.norm(noise, dim=-1, keepdim=True) * attack_level
                x = sample.repeat(batch_size, 1)
                qq = x + noise
            elif attack == "pgd":
                qq = []

                sample.requires_grad = True
                assert sample.requires_grad
                assert sample.is_leaf

                for _ in range(batch_size):
                    pert_sample = pgd(model, sample, model(sample), criterion=torch.nn.CosineSimilarity(dim=-1),
                                      eps=0.002, eps_step=0.002, clip_min=-1, clip_max=1, targeted=False, device=device)
                    qq.append(sample + (pert_sample - sample) / torch.norm(pert_sample - sample) * attack_level)
                qq = torch.stack(qq)
            elif attack == "anonymization":
                path_to_anon_noise = # path to UAP
                anon_noise, _ = torchaudio.load(path_to_anon_noise)
                anon_noise = anon_noise.to(device)
                anon_noise = attack_level * anon_noise / anon_noise.norm()
                
                x_len = sample.shape[-1]
                
                noise = anon_noise.repeat(batch_size, math.ceil(x_len / len(anon_noise)))
                noise = noise[:, :x_len]
                
                x = sample.repeat(batch_size, 1)
                qq = x + noise
                
            with torch.no_grad():
                batch_correct = 1
                for i in range(batch_size):
                    x = qq[i]

                    pred_class = predict_without_guarantee(
                        args,
                        smoothed_model,
                        sample=x,
                        centroids=centroids.to(device),
                        centroid_target=torch.arange(args.classes_per_it_val)
                    )
                    is_correct = int(pred_class == gt_class)
                    batch_correct *= is_correct

                    if not batch_correct:
                        break

                results[k] = results[k] * batch_correct
                results2[k, j] = batch_correct
                if not batch_correct:
                    break

    era = torch.mean(results).item()
    return results, results2, era


def predict_with_radius(args, smoothed_model, sample: torch.tensor, centroids: torch.tensor,
                        centroid_target: torch.tensor):
    '''
    Predict of smoothed model on sample (or abstain), certified radius on sample and time per sample
    '''
    before_time = time()

    pred = smoothed_model.predict(args, sample, centroids, centroid_target)
    # print(f"PREDICITION {pred}")

    after_time = time()
    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

    if pred != -1.0:
        pred_class, adv_class = pred[0][0].cpu().item(), pred[0][1].cpu().item()
        pred_centroid, adv_centroid = pred[1][0], pred[1][1]
        # print("c1 and c2 norms", torch.norm(pred_centroid, p=2, dim=-1, keepdim=True), torch.norm(adv_centroid, p=2, dim=-1, keepdim=True))
        n_samples = pred[2]

        # print("pred and adv centroid", pred_centroid.shape, adv_centroid.shape)
        ## PROBABLY n_samples=args.N but n_samples=n_samples seems to be more reliable. I don't know what is m_value, however in fact they are probably the same
        smoothed_embedding = smoothed_model._sample_smoothed(sample, m_values=1, n_samples=n_samples,
                                                             batch_size=args.batch).mean(dim=1)
        # do not normalize smoothed_embedding

        gamma_lcb, radius, radius_as_in_article = smoothed_model.certified_radius(sample, pred_centroid, adv_centroid,
                                                                                  g_x=smoothed_embedding)
        gamma_lcb = gamma_lcb.cpu().item()
        radius_as_in_article = radius_as_in_article.item()
        radius = radius.item()

    else:
        pred_class = pred
        radius = -1.0
        gamma_lcb = 0.0
        n_samples = args.N * args.K
        radius_as_in_article = -1
        pred_centroid, adv_centroid = None, None

    return pred_class, gamma_lcb, radius, time_elapsed, n_samples, radius_as_in_article, pred_centroid, adv_centroid


def predict_without_guarantee(args, smoothed_model, sample: torch.tensor, centroids: torch.tensor, centroid_target: torch.tensor):
    before_time = time()

    pred = smoothed_model.predict_without_abstain(args, sample, centroids, centroid_target)
    # print(f"PREDICITION {pred}")

    after_time = time()
    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

    pred_class, _ = pred[0][0].cpu().item(), pred[0][1].cpu().item()
    # pred_centroid, adv_centroid = pred[1][0], pred[1][1]
    # print("c1 and c2 norms", torch.norm(pred_centroid, p=2, dim=-1, keepdim=True), torch.norm(adv_centroid, p=2, dim=-1, keepdim=True))
    # n_samples = pred[2]

    # print("pred and adv centroid", pred_centroid.shape, adv_centroid.shape)
    ## PROBABLY n_samples=args.N but n_samples=n_samples seems to be more reliable. I don't know what is m_value, however in fact they are probably the same
    # smoothed_embedding = smoothed_model._sample_smoothed(sample, m_values=1, n_samples=n_samples,
    #                                                         batch_size=args.batch).mean(dim=1)
    # # do not normalize smoothed_embedding

    # gamma_lcb, radius, radius_as_in_article = smoothed_model.certified_radius(sample, pred_centroid, adv_centroid,
    #                                                                             g_x=smoothed_embedding)
    # gamma_lcb = gamma_lcb.cpu().item()
    # radius_as_in_article = radius_as_in_article.item()
    # radius = radius.item()
    return pred_class


class CustomModel(torch.nn.Module):

    def __init__(self, args):
        super(CustomModel, self).__init__()

        self.model_name = args.model_name
        self.device = args.device
        self.emb_size = args.emb_size

        if self.model_name == "pyannote":
            self.model = Model.from_pretrained("pyannote/embedding",
                                               use_auth_token="HF_TOKEN")
        elif self.model_name == "ecapa-tdnn":
            self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                        run_opts={"device": self.device})
        elif self.model_name == "wavlm":
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
            self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(self.device)
        elif self.model_name == "wespeaker":  # resnet-based voxceleb_resnet221_LM.tar.gz
            self.model = wespeaker.load_model('english')
        elif self.model_name == "campplus":  # resnet-based voxceleb_resnet221_LM.tar.gz
            self.model = wespeaker.load_model('campplus')
            self.model.model = self.model.model.eval()
        elif self.model_name == "eres2net":
            self.model = wespeaker.load_model('eres2net')
        else:
            raise NotImplementedError

        if self.model_name not in ["wespeaker", "campplus", "eres2net"]:
            self.model = self.model.to(self.device)
            self.model = self.model.eval()
        else:
            self.model.device = self.device
            self.model.set_gpu(int(args.cuda_number))

    def forward(self, x):
        # x: [BS, T]
        if self.model_name == "wavlm":
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = self.feature_extractor(x, sampling_rate=16000, padding=True, return_tensors="pt")
            x = x["input_values"].squeeze(0)
            x = self.model(x.to(self.device)).embeddings
        elif self.model_name == "pyannote":
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)
            x = self.model(x)
        elif self.model_name == "ecapa-tdnn":
            x = self.model.encode_batch(x)

        elif self.model_name in ["wespeaker", "campplus", "eres2net"]:
            self.model.model = self.model.model.eval()
            embeddings = torch.empty((1, self.emb_size)).to(self.device)
            with torch.no_grad():
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
                for sample in x:
                    sample = sample.unsqueeze(0)
                    sample = self.model.compute_fbank(sample, sample_rate=self.model.resample_rate, cmn=True)
                    sample = sample.unsqueeze(0)
                    sample = sample.to(self.model.device)
                    outputs = self.model.model(sample)
                    outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
                    embed = outputs[0]  # .to(torch.device('cpu'))
                    # print(embeddings.shape, embed.shape)
                    embeddings = torch.cat((embeddings, embed.unsqueeze(0)), dim=0)
            x = embeddings[1:]
        x = x.reshape(-1, 1, self.emb_size)
        x = torch.nn.functional.normalize(x, p=2.0, dim=-1, eps=1e-12, out=None)
        return x  # [BS, 1, EMB_DIM]


class Producer():
    def __init__(self, model, args, normalize, audio_len=3):

        self.device = args.device
        max_num_classes = args.classes_per_it_val
        self.model = model.to(self.device)

        self.shots = args.num_support_val
        self.normalize = normalize
        self.audio_len = audio_len
        
        self.embedding_size = args.emb_size

        speakers_train = []

        self.path_to_data = os.path.join(args.dataset_test, "wav/")
        speakers_test = [idx[-7:] for idx in glob.glob(self.path_to_data + '*')]
        if args.dataset_train:
            self.path_to_train_data = os.path.join(args.dataset_train, "wav/")
            speakers_train = [idx[-7:] for idx in glob.glob(self.path_to_train_data + '*')]

        speakers_paths_train = [glob.glob(self.path_to_train_data + id_ + '/**/*.wav', recursive=True) for id_ in
                                speakers_train]
        speakers_paths_test = [glob.glob(self.path_to_data + id_ + '/**/*.wav', recursive=True) for id_ in
                               speakers_test]

        np.random.seed(42)
        speakers_train = np.array(speakers_train)
        speakers_test = np.array(speakers_test)
        idx1 = np.random.permutation(np.arange(len(speakers_train)))
        idx2 = np.random.permutation(np.arange(len(speakers_test)))
        speakers_train = speakers_train[idx1]
        speakers_paths_train = [speakers_paths_train[idx] for idx in idx1]
        speakers_test = speakers_test[idx2]
        self.speakers_test_only = speakers_test
        speakers_paths_test = [speakers_paths_test[idx] for idx in idx2]

        self.te_pths = speakers_paths_test

        speakers = np.append(speakers_train, speakers_test)
        speakers_paths = speakers_paths_train + speakers_paths_test
        self.id_list = speakers
        self.speakers_paths = speakers_paths
        # restriction on number of classes
        if max_num_classes is not None:
            self.id_list = self.id_list[-max_num_classes:]
            self.speakers_paths = self.speakers_paths[-max_num_classes:]
            n_test_speakers = min(len(self.speakers_test_only), max_num_classes)
            self.speakers_test_only = self.speakers_test_only[-n_test_speakers:]
        self.num_classes = len(self.id_list)

        # evaluating class prototypes
        self.class_prototypes = dict.fromkeys(self.id_list)
        self._calculate_class_prototypes()

    def _calculate_class_prototypes(self):
        """
        Computes the mean embedding (prototype) for each class (speaker).

        This method iterates through the list of speaker IDs, processes their corresponding audio files
        to obtain embeddings, and then calculates the mean embedding for each speaker. The mean embeddings
        are stored in the `self.class_prototypes` dictionary, with speaker IDs as keys.

        Note:
            The model and the device should be set before calling this method. The method assumes that
            `self.model.audio()` is a method that processes an audio file path to return a tensor suitable
            for embedding computation, and `self.model()` is a method that computes the embedding given
            an input tensor. True for Pyannote models.
        """
        self.model.eval()

        self.warning_speakers = []
        self.speaker_enrollment_audios = dict.fromkeys(self.id_list)
        self.speaker_inference_audios = dict.fromkeys(self.id_list)

        with torch.no_grad():
            for speaker_id, speaker_path in tqdm(zip(self.id_list, self.speakers_paths), total=len(self.id_list)):
                # selecting random subset of speaker audios for prototype constructing
                if self.shots is None:
                    speaker_enrollment_subset = speaker_path
                elif self.shots > len(speaker_path):
                    # will call a warninig that speaker has less audios then shots
                    self.warning_speakers.append(speaker_id)
                    # selecting all existing audios
                    speaker_enrollment_subset = speaker_path
                else:
                    speaker_enrollment_subset = random.sample(speaker_path, self.shots)
                self.speaker_enrollment_audios[speaker_id] = speaker_enrollment_subset

                # saving unused audios for test dataset
                self.speaker_inference_audios[speaker_id] = [path for path in speaker_path if
                                                             path not in speaker_enrollment_subset]

                # reading audios from pathes
                # audios_map = map(lambda x: self.model.model.audio(x)[0][:, 0:3*16000].to(self.device), speaker_enrollment_subset)
                # audios_map = map(lambda x: torchaudio.load(x)[0][:, 0:3*16000].to(self.device), speaker_enrollment_subset)

                # if self.model.model_name == "wavlm":
                #     audios_map = [torchaudio.load(x)[0][0, 0:3*16000].squeeze(0).to(self.device) for x in speaker_enrollment_subset]
                # else:
                #     audios_map = [torchaudio.load(x)[0][:, 0:3*16000].to(self.device) for x in speaker_enrollment_subset]
                audios_map = [torchaudio.load(x)[0][:, 0:self.audio_len * 16000].to(self.device) for x in
                              speaker_enrollment_subset]

                # print("Debug: Tensor device check:", audios_map[0].device)
                # print("Debug: Model device check:", self.model.device)
                # applying model to audios
                # mean_embedding = torch.stack(tuple(map(lambda x: self.model(x), audios_map))).mean(dim=0)
                mean_embedding = torch.stack([self.model(x) for x in audios_map]).mean(dim=0)
                mean_embedding = mean_embedding.reshape(1, -1)

                #### AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                if self.normalize:
                    mean_embedding /= torch.norm(mean_embedding, p=2, dim=-1, keepdim=True)

                #### AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                self.class_prototypes[speaker_id] = mean_embedding

        # warninig
        if len(self.warning_speakers) != 0:
            print(f"Speakers: {self.warning_speakers} have less audios then shots!")
        self.speaker_id_to_class = {}
        self.class_to_speaker_id = {}
        for i, speaker in enumerate(self.speaker_enrollment_audios):
            self.speaker_id_to_class[speaker] = i
            self.class_to_speaker_id[i] = speaker

    def produce_subsets(self):
        return self.class_prototypes, self.speaker_enrollment_audios, self.speaker_inference_audios, self.speaker_id_to_class, self.class_to_speaker_id


def pgd(model: torch.nn.Module, audio: torch.Tensor, target_emb: torch.Tensor, num_iter: int = 10,
        *, criterion, eps: float, eps_step: float, clip_min: float, clip_max: float, targeted: bool = True,
        device='cuda') -> torch.Tensor:
    """
    Applies the Projected Gradient Descent (PGD) attack on the input audio to generate an adversarial audio.

    Parameters:
    - audio (torch.Tensor): The input audio to be perturbed
    - target (torch.Tensor): The target tensor for the attack (e.g., the target class embedding)
    - num_iter (int, optional): The number of iterations for the PGD attack (default is 10)
    - criterion: The loss criterion used for the adversarial perturbation
    - eps (float): The maximum perturbation allowed for each audio in the input
    - eps_step (float): The step size of each iteration in the PGD attack
    - clip_min (float): The minimum value to which the perturbed audio is clipped
    - clip_max (float): The maximum value to which the perturbed audio is clipped
    - targeted (bool, optional): True if the attack is targeted, otherwise False

    Returns:
    - torch.Tensor: The adversarial audio generated by the PGD attack.
    """
    model.eval()
    audio = torch.autograd.Variable(audio + torch.FloatTensor(audio.shape).uniform_(-eps, eps).to(device),
                                    requires_grad=True)

    for _ in range(num_iter):
        # Calculating grad
        emb_audio = model(audio)
        loss = criterion(emb_audio, target_emb)
        model.zero_grad()
        loss.backward(retain_graph=True)

        # FGSM
        grad = audio.grad.data
        pert_audio = fgsm(audio, grad, eps_step, targeted=targeted)

        # Projection
        pert_audio.clamp_(clip_min, clip_max)

        audio = pert_audio.clone().detach().requires_grad_()

    return audio


def fgsm(audio: torch.Tensor, grad: torch.Tensor, epsilon: float, targeted: bool = False) -> torch.Tensor:
    """
    The FGSM (Gradient Sign Method) attack

    Arguments:
    — audio (torch.Tensor): input audio
    — grad (torch.Tensor): gradient of the loss function with respect to the given audio
    — epsilon (float): parameter determining the magnitude of added noise
    — targeted (bool, optional): True if the attack is targeted, otherwise False

    Returns:
    — pert_audio (torch.Tensor): modified audio as a result of the attack
    """

    if targeted:
        pert_audio = audio + epsilon * grad.sign()
    else:
        pert_audio = audio - epsilon * grad.sign()

    return pert_audio


def samples_count(model, audio, N, centroids, args, device):
    batch_size = args.batch
    model.eval()

    with torch.no_grad():
        counts = np.zeros(args.classes_per_it_val, dtype=int)
        for _ in range(math.ceil(N / batch_size)):
            this_batch_size = min(batch_size, N)
            N -= this_batch_size

            batch = audio.repeat((this_batch_size, 1))
            noise = torch.randn_like(batch, device=device) * args.sigma
            qq = (batch + noise).squeeze(1)
            pred_emb = model(qq)

            predictions = (pred_emb @ centroids.T).argmax(dim=-1).cpu()

            for idx in predictions:
                counts[idx] += 1

    return counts


def predict_with_radius_2(args, model, sample, centroids, centroid_target):
    device = args.device
    N0 = args.N
    N = args.N * args.K
    alpha = args.alpha
    sigma = args.sigma

    counts_selection = samples_count(model, sample, N0, centroids, args, device)
    cAHat = counts_selection.argmax().item()

    counts_estimation = samples_count(model, sample, N, centroids, args, device)
    nA = counts_estimation[cAHat]

    pABar = proportion_confint(nA, N, alpha=2 * alpha, method='beta')[0]

    if pABar < 0.5:
        return centroid_target[cAHat], -1  # Abstain
    else:
        return centroid_target[cAHat], sigma * norm.ppf(pABar)


def predict_with_radius_3(args, smoothed_model, sample, centroids, centroid_target):
    n_samples = args.N
    batch = args.batch

    g = smoothed_model._sample_smoothed(sample, m_values=1, n_samples=n_samples, batch_size=batch).mean(dim=1)
    g = g / g.norm()

    pred = smoothed_model.predict(args, sample, centroids, centroid_target)
    if pred != -1.0:
        pred_class, adv_class = pred[0][0].cpu().item(), pred[0][1].cpu().item()
        pred_centroid, adv_centroid = pred[1][0], pred[1][1]

        C1dotC2 = pred_centroid @ adv_centroid
        C1plucC2norm = (pred_centroid + adv_centroid).norm()

        alpha_ = ((1 + C1dotC2) / (2 * C1plucC2norm) - 1) / (((1 + C1dotC2) / (C1plucC2norm)) ** 2 - 1)
        beta_ = ((1 + C1dotC2) / C1plucC2norm - 0.5) / (((1 + C1dotC2) / (C1plucC2norm)) ** 2 - 1)

        delta = alpha_ * (g @ pred_centroid) + beta_ * (g @ ((pred_centroid + adv_centroid) / C1plucC2norm))
        radius = args.sigma * norm.ppf(delta.cpu().item())
    else:
        pred_class = pred
        radius = -1.0

    return pred_class, radius


def predict_with_radius_4(args, smoothed_model, sample, centroids, centroid_target):
    n_samples = args.N
    batch = args.batch

    g = smoothed_model._sample_smoothed(sample, m_values=1, n_samples=n_samples, batch_size=batch).mean(dim=1)
    pred = smoothed_model.predict(args, sample, centroids, centroid_target)
    if pred != -1.0:
        pred_class, adv_class = pred[0][0].cpu().item(), pred[0][1].cpu().item()
        pred_centroid, adv_centroid = pred[1][0], pred[1][1]
        c1 = pred_centroid
        c2 = adv_centroid
        # v4
        # dot = c1 @ c2
        # denom = c1 @ c1 * c2 @ c2 - dot.pow(2)
        # alpha = (dot + c2 @ c2) / denom
        # beta = -2 * dot / denom
        # delta = alpha * (g @ c1) + beta * (g @ (c1 + c2) / 2)

        # v5
        # alpha = (c1 @ c2 + c2 @ c2) / 2 / (c1@c1 * c2@c2 - (c1@c2) ** 2)
        # beta = (-c1 @ c2 + c2 @ c2) / (c1 @ c1 * c2 @ c2 - (c1@c2) ** 2)
        # delta = alpha * g @ (c1 - c2) + beta * (g @ (c1 + c2) / 2)

        # v6
        # g = g / g.norm()
        # alpha = (c2 @ c2) / (c1 @ c1 * c2 @ c2 - (c1@c2) ** 2)
        # beta = (-c1 @ c2) / (c1 @ c1 * c2 @ c2 - (c1@c2) ** 2)
        # delta = alpha * g @ c1 + beta * g @ c2

        # v7
        # alpha = 1 / (c1 @ c1 -2 * c1 @ c2 + c2 @ c2)
        # beta = (-c1@c2 + c2@c2) / (c1@c1 - 2*c1@c2 + c2@c2)

        # delta = alpha * g @ (c1 - c2) + beta
        delta = 1 - torch.norm(g - c1) / 2 / torch.norm(c1 - c2)
        print(g @ c1, g @ c2, delta)
        radius = args.sigma * norm.ppf(delta.cpu().item())
    else:
        pred_class = pred
        radius = -1.0

    return pred_class, radius
