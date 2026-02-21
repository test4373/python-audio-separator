"""
models_config.py - Extended model configuration for SESA HuggingFace Space

Provides:
1. EXTENDED_MODELS: All models from the main SESA project merged with base ROFORMER_MODELS
2. Custom model management (add/delete/list from custom_models.json)
3. Audio segmentation helpers for large file handling
4. Batch processing support
"""

import os
import json
import subprocess
import logging
import re
import shutil
import requests
import urllib.parse

logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_MODELS_FILE = os.path.join(BASE_DIR, 'custom_models.json')
MODEL_CACHE_DIR = "/tmp/audio-separator-models/"

# ─── Segmentation Constants ─────────────────────────────────────────────────
MAX_UNSPLIT_DURATION = 1800   # 30 minutes
SEGMENT_DURATION = 600       # 10 minutes per segment

# ─── Extended Model Registry ────────────────────────────────────────────────
# Merged from main project model.py + existing ROFORMER_MODELS
# Format: { "Category": { "Display Name": "checkpoint_filename.ckpt" } }
EXTENDED_MODELS = {
    "Vocals": {
        # === From main project model.py ===
        'BS Roformer HyperACEv2 Voc (by unwa)': 'bs_roformer_voc_hyperacev2.ckpt',
        'BS-Roformer-Resurrection (by unwa)': 'BS-Roformer-Resurrection.ckpt',
        'BS Roformer Revive3e (by unwa)': 'bs_roformer_revive3e.ckpt',
        'BS Roformer Revive2 (by unwa)': 'bs_roformer_revive2.ckpt',
        'BS Roformer Revive (by unwa)': 'bs_roformer_revive.ckpt',
        'Karaoke BS Roformer (by anvuew)': 'karaoke_bs_roformer_anvuew.ckpt',
        'VOCALS big_beta6X (by Unwa)': 'big_beta6x.ckpt',
        'VOCALS big_beta6 (by Unwa)': 'big_beta6.ckpt',
        'VOCALS Mel-Roformer FT 3 Preview (by unwa)': 'kimmel_unwa_ft3_prev.ckpt',
        'VOCALS InstVocHQ (MDX23C)': 'model_vocals_mdx23c_sdr_10.17.ckpt',
        'VOCALS MelBand-Roformer (by KimberleyJSN)': 'MelBandRoformer.ckpt',
        'VOCALS BS-Roformer 1297 (by viperx)': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        'VOCALS BS-Roformer 1296 (by viperx)': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
        'VOCALS BS-RoformerLargeV1 (by unwa)': 'BS-Roformer_LargeV1.ckpt',
        'VOCALS Mel-Roformer big beta 4 (by unwa)': 'melband_roformer_big_beta4.ckpt',
        'VOCALS Melband-Roformer BigBeta5e (by unwa)': 'big_beta5e.ckpt',
        'VOCALS VitLarge23 (by ZFTurbo)': 'model_vocals_segm_models_sdr_9.77.ckpt',
        'VOCALS MelBand-Roformer Kim FT (by Unwa)': 'kimmel_unwa_ft.ckpt',
        'VOCALS MelBand-Roformer Kim FT 2 (by Unwa)': 'kimmel_unwa_ft2.ckpt',
        'VOCALS MelBand-Roformer Kim FT 2 Bleedless (by unwa)': 'kimmel_unwa_ft2_bleedless.ckpt',
        'VOCALS MelBand-Roformer (by Becruily)': 'mel_band_roformer_vocals_becruily.ckpt',
        'VOCALS Male-Female BS-RoFormer 7.2889 (by aufr33)': 'bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt',
        'voc_gaboxBSroformer (by Gabox)': 'voc_gaboxBSR.ckpt',
        'voc_gaboxMelReformer (by Gabox)': 'voc_gabox.ckpt',
        'Voc FV3 (by Gabox)': 'voc_Fv3.ckpt',
        'FullnessVocalModel (by Amane)': 'FullnessVocalModel.ckpt',
        'voc_fv4 (by Gabox)': 'voc_fv4.ckpt',
        'voc_fv5 (by Gabox)': 'voc_fv5.ckpt',
        'voc_fv6 (by Gabox)': 'voc_fv6.ckpt',
        'voc_fv7 (by Gabox)': 'voc_fv7.ckpt',
        'vocfv7beta1 (by Gabox)': 'vocfv7beta1.ckpt',
        'vocfv7beta2 (by Gabox)': 'vocfv7beta2.ckpt',
        'vocfv7beta3 (by Gabox)': 'vocfv7beta3.ckpt',
        'MelBandRoformerSYHFTV3Epsilon (by SYH99999)': 'MelBandRoformerSYHFTV3Epsilon.ckpt',
        'MelBandRoformerBigSYHFTV1 (by SYH99999)': 'MelBandRoformerBigSYHFTV1.ckpt',
        'model_chorus_bs_roformer_ep_146 (by Sucial)': 'model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt',
        'model_chorus_bs_roformer_ep_267 (by Sucial)': 'model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt',
        'BS-Rofo-SW-Fixed (by jarredou)': 'BS-Rofo-SW-Fixed.ckpt',
        'BS_ResurrectioN (by Gabox)': 'BS_ResurrectioN.ckpt',
        # === Original ROFORMER_MODELS ===
        'MelBand Roformer | Big Beta 6X by unwa': 'melband_roformer_big_beta6x.ckpt',
        'MelBand Roformer Kim | Big Beta 4 FT by unwa': 'melband_roformer_big_beta4.ckpt',
        'MelBand Roformer | Kim FT by unwa': 'kimmel_unwa_ft.ckpt',
        'MelBand Roformer | Kim FT 2 by unwa': 'kimmel_unwa_ft2.ckpt',
        'MelBand Roformer | Kim FT 2 Bleedless by unwa': 'kimmel_unwa_ft2_bleedless.ckpt',
        'MelBand Roformer | Kim FT 3 Preview by unwa': 'kimmel_unwa_ft3_prev.ckpt',
        'MelBand Roformer | Vocals FV1 by Gabox': 'mel_band_roformer_vocals_fv1_gabox.ckpt',
        'MelBand Roformer | Vocals FV2 by Gabox': 'mel_band_roformer_vocals_fv2_gabox.ckpt',
        'MelBand Roformer | Vocals FV3 by Gabox': 'mel_band_roformer_vocals_fv3_gabox.ckpt',
        'MelBand Roformer | Vocals FV4 by Gabox': 'mel_band_roformer_vocals_fv4_gabox.ckpt',
        'BS Roformer | Chorus Male-Female by Sucial': 'model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt',
        'BS Roformer | Male-Female by aufr33': 'bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt',
    },
    "Instrumentals": {
        # === From main project ===
        'Neo_InstVFX (by natanworkspace)': 'Neo_InstVFX.ckpt',
        'BS-Roformer-Resurrection-Inst (by unwa)': 'BS-Roformer-Resurrection-Inst.ckpt',
        'BS Roformer HyperACEv2 Inst (by unwa)': 'bs_roformer_inst_hyperacev2.ckpt',
        'BS-Roformer-Large-Inst (by unwa)': 'bs_large_v2_inst.ckpt',
        'BS Roformer FNO (by unwa)': 'bs_roformer_fno.ckpt',
        'Rifforge final 14.24 (by meskvlla33)': 'rifforge_full_sdr_14.2436.ckpt',
        'Inst_GaboxFv8 (by Gabox)': 'Inst_GaboxFv8.ckpt',
        'Inst_GaboxFv9 (by Gabox)': 'Inst_GaboxFv9.ckpt',
        'inst_gaboxFlowersV10 (by Gabox)': 'inst_gaboxFlowersV10.ckpt',
        'INST Mel-Roformer v1 (by unwa)': 'melband_roformer_inst_v1.ckpt',
        'INST Mel-Roformer v1e+ (by unwa)': 'inst_v1e_plus.ckpt',
        'INST Mel-Roformer v1+ (by unwa)': 'inst_v1_plus_test.ckpt',
        'INST Mel-Roformer v2 (by unwa)': 'melband_roformer_inst_v2.ckpt',
        'INST MelBand-Roformer (by Becruily)': 'mel_band_roformer_instrumental_becruily.ckpt',
        'inst_v1e (by unwa)': 'inst_v1e.ckpt',
        'inst_gabox (by Gabox)': 'inst_gabox.ckpt',
        'inst_gaboxBV1 (by Gabox)': 'inst_gaboxBv1.ckpt',
        'inst_gaboxBV2 (by Gabox)': 'inst_gaboxBv2.ckpt',
        'inst_gaboxFV2 (by Gabox)': 'inst_gaboxFv2.ckpt',
        'inst_Fv3 (by Gabox)': 'inst_gaboxFv3.ckpt',
        'Intrumental_Gabox (by Gabox)': 'intrumental_gabox.ckpt',
        'inst_Fv4Noise (by Gabox)': 'inst_Fv4Noise.ckpt',
        'inst_Fv4 (by Gabox)': 'inst_Fv4.ckpt',
        'INSTV5 (by Gabox)': 'INSTV5.ckpt',
        'INSTV5N (by Gabox)': 'INSTV5N.ckpt',
        'INSTV6N (by Gabox)': 'INSTV6N.ckpt',
        'Inst_GaboxV7 (by Gabox)': 'Inst_GaboxV7.ckpt',
        'INSTV7N (by Gabox)': 'INSTV7N.ckpt',
        'inst_fv7b (by Gabox)': 'inst_fv7b.ckpt',
        'inst_fv7z (by Gabox)': 'Inst_GaboxFv7z.ckpt',
        'Inst_FV8b (by Gabox)': 'Inst_FV8b.ckpt',
        # === Original ROFORMER_MODELS ===
        'MelBand Roformer | FVX by Gabox': 'mel_band_roformer_instrumental_fvx_gabox.ckpt',
        'MelBand Roformer | INSTV8N by Gabox': 'mel_band_roformer_instrumental_instv8n_gabox.ckpt',
        'MelBand Roformer | Instrumental V1 by unwa': 'melband_roformer_inst_v1.ckpt',
        'MelBand Roformer | Instrumental V2 by unwa': 'melband_roformer_inst_v2.ckpt',
        'MelBand Roformer | Instrumental 2 by Gabox': 'mel_band_roformer_instrumental_2_gabox.ckpt',
        'MelBand Roformer | Instrumental 3 by Gabox': 'mel_band_roformer_instrumental_3_gabox.ckpt',
        'MelBand Roformer | Instrumental Bleedless V1 by Gabox': 'mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt',
        'MelBand Roformer | Instrumental Bleedless V2 by Gabox': 'mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt',
        'MelBand Roformer | Inst V1E by unwa': 'inst_v1e.ckpt',
        'MelBand Roformer | Inst V1E+ by unwa': 'inst_v1e_plus.ckpt',
        'MelBand Roformer | INSTV5N by Gabox': 'mel_band_roformer_instrumental_instv5n_gabox.ckpt',
        'MelBand Roformer | INSTV6N by Gabox': 'mel_band_roformer_instrumental_instv6n_gabox.ckpt',
        'MelBand Roformer | INSTV7 by Gabox': 'mel_band_roformer_instrumental_instv7_gabox.ckpt',
    },
    "InstVoc Duality": {
        'MelBand Roformer Kim | InstVoc Duality V1 by Unwa': 'melband_roformer_instvoc_duality_v1.ckpt',
        'MelBand Roformer Kim | InstVoc Duality V2 by Unwa': 'melband_roformer_instvox_duality_v2.ckpt',
        'INST-VOC Duality v1 (by unwa)': 'melband_roformer_instvoc_duality_v1.ckpt',
        'INST-VOC Duality v2 (by unwa)': 'melband_roformer_instvox_duality_v2.ckpt',
    },
    "De-Reverb": {
        'DE-REVERB MDX23C (by aufr33 & jarredou)': 'dereverb_mdx23c_sdr_6.9096.ckpt',
        'DE-REVERB MelBand-Roformer 19.1729 (by anvuew)': 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
        'DE-REVERB-Echo MelBand-Roformer (by Sucial)': 'dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt',
        'dereverb_less_aggressive (by anvuew)': 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt',
        'dereverb_mono (by anvuew)': 'dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt',
        'dereverb-echo 128_4_4 (by Sucial)': 'dereverb-echo_128_4_4_mel_band_roformer_sdr_dry_12.4235.ckpt',
        'dereverb_echo_mbr_v2 (by Sucial)': 'dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt',
        'de_big_reverb_mbr_ep_362 (by Sucial)': 'de_big_reverb_mbr_ep_362.ckpt',
        'de_super_big_reverb_mbr_ep_346 (by Sucial)': 'de_super_big_reverb_mbr_ep_346.ckpt',
        'dereverb_room (by anvuew)': 'dereverb_room_anvuew_sdr_13.7432.ckpt',
        # === Original ROFORMER_MODELS ===
        'MelBand Roformer | De-Reverb by anvuew': 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
        'MelBand Roformer | De-Reverb Less Aggressive by anvuew': 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt',
        'MelBand Roformer | De-Reverb-Echo V2 by Sucial': 'dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt',
        'MelBand Roformer | De-Reverb-Echo Fused by Sucial': 'dereverb_echo_mbr_fused.ckpt',
    },
    "Denoise": {
        'DENOISE MelBand-Roformer-1 (by aufr33)': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
        'DENOISE MelBand-Roformer-2 aggr (by aufr33)': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
        'denoisedebleed (by Gabox)': 'denoisedebleed.ckpt',
        'bleed_suppressor_v1 (by unwa)': 'bleed_suppressor_v1.ckpt',
        # === Original ROFORMER_MODELS ===
        'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
        'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
        'MelBand Roformer | Denoise by Gabox': 'mel_band_roformer_denoise_gabox.ckpt',
        'MelBand Roformer | Karaoke by Gabox': 'mel_band_roformer_karaoke_gabox.ckpt',
        'MelBand Roformer | Karaoke by becruily': 'mel_band_roformer_karaoke_becruily.ckpt',
    },
    "Karaoke": {
        'KARAOKE MelBand-Roformer (by aufr33 & viperx)': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
        'KaraokeGabox (by Gabox)': 'Karaoke_GaboxV1.ckpt',
        'bs_karaoke_gabox_IS (by Gabox)': 'bs_karaoke_gabox_IS.ckpt',
        'bs_roformer_karaoke_frazer_becruily': 'bs_roformer_karaoke_frazer_becruily.ckpt',
        'mel_band_roformer_karaoke_becruily': 'mel_band_roformer_karaoke_becruily.ckpt',
    },
    "4-Stem": {
        '4STEMS SCNet MUSDB18 (by starrytong)': 'scnet_checkpoint_musdb18.ckpt',
        '4STEMS SCNet XL MUSDB18 (by ZFTurbo)': 'model_scnet_ep_54_sdr_9.8051.ckpt',
        '4STEMS SCNet Large (by starrytong)': 'SCNet-large_starrytong_fixed.ckpt',
        '4STEMS BS-Roformer MUSDB18 (by ZFTurbo)': 'model_bs_roformer_ep_17_sdr_9.6568.ckpt',
        'MelBandRoformer4StemFTLarge (by SYH99999)': 'MelBandRoformer4StemFTLarge.ckpt',
    },
    "General Purpose": {
        # === From main project ===
        'OTHER BS-Roformer 1053 (by viperx)': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
        'CROWD-REMOVAL MelBand-Roformer (by aufr33)': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
        'CINEMATIC BandIt Plus (by kwatcharasupat)': 'model_bandit_plus_dnr_sdr_11.47.chpt',
        'CINEMATIC BandIt v2 multi (by kwatcharasupat)': 'checkpoint-multi_state_dict.ckpt',
        'DRUMSEP MDX23C 6stem (by aufr33 & jarredou)': 'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt',
        'bs_hyperace (by unwa)': 'bs_hyperace.ckpt',
        'becruily_deux (by becruily)': 'becruily_deux.ckpt',
        'becruily_guitar (by becruily)': 'becruily_guitar.ckpt',
        'aspiration MelBand-Roformer (by Sucial)': 'aspiration_mel_band_roformer_sdr_18.9845.ckpt',
        'mdx23c_similarity (by ZFTurbo)': 'model_mdx23c_ep_271_l1_freq_72.2383.ckpt',
        'Lead_Rhythm_Guitar (by listra92)': 'model_mel_band_roformer_ep_72_sdr_3.2232.ckpt',
        'last_bs_roformer_4stem (by Amane)': 'last_bs_roformer.ckpt',
        'bs_roformer_4stems_ft (by SYH99999)': 'bs_roformer_4stems_ft.pth',
        'CINEMATIC BandIt v2 Eng (by kwatcharasupat)': 'checkpoint-eng_state_dict.ckpt',
        # === Original ROFORMER_MODELS ===
        'BS-Roformer-Viperx-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        'BS-Roformer-Viperx-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
        'BS-Roformer-De-Reverb': 'deverb_bs_roformer_8_384dim_10depth.ckpt',
        'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
        'MelBand Roformer | Crowd Removal by aufr33': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
        'MelBand Roformer | Aspiration by Sucial': 'aspiration_mel_band_roformer_sdr_18.9845.ckpt',
        'MelBand Roformer | Aspiration Less Aggressive by Sucial': 'aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt',
    }
}


# ─── Model Download URLs ────────────────────────────────────────────────────
# Maps checkpoint filenames to their download URLs (config + checkpoint + optional .py)
# Extracted from main SESA project's MODEL_CONFIGS in model.py
MODEL_DOWNLOAD_URLS = {
    # === Vocal Models ===
    'big_beta6x.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.ckpt'],
    'big_beta6.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.ckpt'],
    'kimmel_unwa_ft3_prev.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-kim/resolve/main/kimmel_unwa_ft3_prev.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-kim/resolve/main/kimmel_unwa_ft3_prev.ckpt'],
    'model_vocals_mdx23c_sdr_10.17.ckpt': ['https://huggingface.co/ASesYusuf1/MODELS/resolve/main/model_2_stem_full_band_8k.yaml', 'https://huggingface.co/ASesYusuf1/MODELS/resolve/main/model_vocals_mdx23c_sdr_10.17.ckpt'],
    'MelBandRoformer.ckpt': ['https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.yaml', 'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt'],
    'model_bs_roformer_ep_317_sdr_12.9755.ckpt': ['https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml', 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt'],
    'model_bs_roformer_ep_368_sdr_12.9628.ckpt': ['https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml', 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt'],
    'BS-Roformer_LargeV1.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-1076/resolve/main/BS-Roformer_LargeV1.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-1076/resolve/main/BS-Roformer_LargeV1.ckpt'],
    'melband_roformer_big_beta4.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt'],
    'big_beta5e.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt'],
    'model_vocals_segm_models_sdr_9.77.ckpt': ['https://huggingface.co/ZFTurbo/MelBandRoformerVitLarge23/resolve/main/config_vocals_segm_models.yaml', 'https://huggingface.co/ZFTurbo/MelBandRoformerVitLarge23/resolve/main/model_vocals_segm_models_sdr_9.77.ckpt'],
    'kimmel_unwa_ft.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-kim/resolve/main/kimmel_unwa_ft.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-kim/resolve/main/kimmel_unwa_ft.ckpt'],
    'kimmel_unwa_ft2.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-kim/resolve/main/kimmel_unwa_ft2.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-kim/resolve/main/kimmel_unwa_ft2.ckpt'],
    'kimmel_unwa_ft2_bleedless.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-kim/resolve/main/kimmel_unwa_ft2.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-kim/resolve/main/kimmel_unwa_ft2_bleedless.ckpt'],
    'mel_band_roformer_vocals_becruily.ckpt': ['https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml', 'https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt'],
    'bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt': ['https://huggingface.co/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.yaml', 'https://huggingface.co/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt'],
    'voc_gaboxBSR.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/bsroformers/bs_roformer_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/bsroformers/voc_gaboxBSR.ckpt'],
    'voc_gabox.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.ckpt'],
    'voc_Fv3.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt'],
    'FullnessVocalModel.ckpt': ['https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/config.yaml', 'https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/FullnessVocalModel.ckpt'],
    'voc_fv4.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv4.ckpt'],
    'voc_fv5.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv5.ckpt'],
    'voc_fv6.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv6.ckpt'],
    'voc_fv7.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv7.ckpt'],
    'vocfv7beta1.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/vocfv7beta1.ckpt'],
    'vocfv7beta2.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/vocfv7beta2.ckpt'],
    'vocfv7beta3.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/vocfv7beta3.ckpt'],
    'MelBandRoformerSYHFTV3Epsilon.ckpt': ['https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/config.yaml', 'https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/MelBandRoformerSYHFTV3Epsilon.ckpt'],
    'MelBandRoformerBigSYHFTV1.ckpt': ['https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1/resolve/main/config.yaml', 'https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1/resolve/main/MelBandRoformerBigSYHFTV1.ckpt'],
    'model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt': ['https://huggingface.co/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/model_chorus_bs_roformer.yaml', 'https://huggingface.co/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt'],
    'model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt': ['https://huggingface.co/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/model_chorus_bs_roformer.yaml', 'https://huggingface.co/Sucial/Chorus_Male_Female_BS_RoFormer/resolve/main/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt'],
    'BS-Rofo-SW-Fixed.ckpt': ['https://huggingface.co/jarredou/BS-Rofo-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml', 'https://huggingface.co/jarredou/BS-Rofo-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt'],
    'bs_roformer_voc_hyperacev2.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-HyperACEv2/resolve/main/bs_roformer_voc_hyperacev2.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-HyperACEv2/resolve/main/bs_roformer_voc_hyperacev2.ckpt'],
    'BS-Roformer-Resurrection.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection.ckpt'],
    'bs_roformer_revive3e.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.ckpt'],
    'bs_roformer_revive2.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive2.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive2.ckpt'],
    'bs_roformer_revive.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive.ckpt'],
    'karaoke_bs_roformer_anvuew.ckpt': ['https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.yaml', 'https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.ckpt'],
    'BS_ResurrectioN.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/bsroformers/BS_ResurrectioN.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/bsroformers/BS_ResurrectioN.ckpt'],
    # === Instrumental Models ===
    'Neo_InstVFX.ckpt': ['https://huggingface.co/naitotomato/Neo_InstVFX/resolve/main/Neo_InstVFX.yaml', 'https://huggingface.co/naitotomato/Neo_InstVFX/resolve/main/Neo_InstVFX.ckpt'],
    'BS-Roformer-Resurrection-Inst.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt'],
    'bs_roformer_inst_hyperacev2.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-HyperACEv2/resolve/main/bs_roformer_inst_hyperacev2.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-HyperACEv2/resolve/main/bs_roformer_inst_hyperacev2.ckpt'],
    'bs_large_v2_inst.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-1076/resolve/main/BS-Roformer_LargeV1.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-1076/resolve/main/bs_large_v2_inst.ckpt'],
    'bs_roformer_fno.ckpt': ['https://huggingface.co/pcunwa/BS-Roformer-FNO/resolve/main/bs_roformer_fno.yaml', 'https://huggingface.co/pcunwa/BS-Roformer-FNO/resolve/main/bs_roformer_fno.ckpt'],
    'rifforge_full_sdr_14.2436.ckpt': ['https://huggingface.co/meskvlla33/rifforge/resolve/main/rifforge_config.yaml', 'https://huggingface.co/meskvlla33/rifforge/resolve/main/rifforge_full_sdr_14.2436.ckpt'],
    'melband_roformer_inst_v1.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt'],
    'inst_v1e_plus.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt'],
    'inst_v1_plus_test.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1_plus_test.ckpt'],
    'melband_roformer_inst_v2.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt'],
    'mel_band_roformer_instrumental_becruily.ckpt': ['https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml', 'https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt'],
    'inst_v1e.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt'],
    'inst_gabox.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.ckpt'],
    'inst_gaboxBv1.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt'],
    'inst_gaboxBv2.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt'],
    'inst_gaboxFv2.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt'],
    'inst_gaboxFv3.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt'],
    'intrumental_gabox.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/intrumental_gabox.ckpt'],
    'inst_Fv4Noise.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt'],
    'inst_Fv4.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4.ckpt'],
    'INSTV5.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt'],
    'INSTV5N.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt'],
    'INSTV6N.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt'],
    'Inst_GaboxV7.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt'],
    'INSTV7N.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV7N.ckpt'],
    'inst_fv7b.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/inst_fv7b.ckpt'],
    'Inst_GaboxFv7z.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv7z.ckpt'],
    'Inst_GaboxFv9.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv9.ckpt'],
    'inst_gaboxFlowersV10.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/v10.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFlowersV10.ckpt'],
    'Inst_FV8b.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/Inst_FV8b.ckpt'],
    'Inst_Fv8.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/Inst_Fv8.ckpt'],
    'Inst_GaboxFv8.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv8.ckpt'],
    'inst_gaboxFv1.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt'],
    'INSTV6.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt'],
    'gaboxFv1.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt'],
    # === InstVoc Duality ===
    'melband_roformer_instvoc_duality_v1.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt'],
    'melband_roformer_instvox_duality_v2.ckpt': ['https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.yaml', 'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt'],
    # === 4-Stem Models ===
    'scnet_checkpoint_musdb18.ckpt': ['https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml', 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt'],
    'model_scnet_ep_54_sdr_9.8051.ckpt': ['https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml', 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt'],
    'SCNet-large_starrytong_fixed.ckpt': ['https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml', 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt'],
    'model_bs_roformer_ep_17_sdr_9.6568.ckpt': ['https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml', 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt'],
    'MelBandRoformer4StemFTLarge.ckpt': ['https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml', 'https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt'],
    # === Denoise Models ===
    'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt': ['https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml', 'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'],
    'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt': ['https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml', 'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'],
    'denoisedebleed.ckpt': ['https://huggingface.co/poiqazwsx/melband-roformer-denoise/resolve/main/model_mel_band_roformer_denoise.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt'],
    'bleed_suppressor_v1.ckpt': ['https://huggingface.co/ASesYusuf1/MODELS/resolve/main/config_bleed_suppressor_v1.yaml', 'https://huggingface.co/ASesYusuf1/MODELS/resolve/main/bleed_suppressor_v1.ckpt'],
    # === Dereverb Models ===
    'dereverb_mdx23c_sdr_6.9096.ckpt': ['https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml', 'https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt'],
    'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt': ['https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml', 'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'],
    'dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt': ['https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml', 'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt'],
    'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt': ['https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml', 'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt'],
    'dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt': ['https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml', 'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'],
    'dereverb-echo_128_4_4_mel_band_roformer_sdr_dry_12.4235.ckpt': ['https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_128_4_4_mel_band_roformer.yaml', 'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_128_4_4_mel_band_roformer_sdr_dry_12.4235.ckpt'],
    'dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt': ['https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml', 'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt'],
    'de_big_reverb_mbr_ep_362.ckpt': ['https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml', 'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_big_reverb_mbr_ep_362.ckpt'],
    'de_super_big_reverb_mbr_ep_346.ckpt': ['https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml', 'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_super_big_reverb_mbr_ep_346.ckpt'],
    'dereverb_room_anvuew_sdr_13.7432.ckpt': ['https://huggingface.co/anvuew/dereverb_room/resolve/main/dereverb_room_anvuew.yaml', 'https://huggingface.co/anvuew/dereverb_room/resolve/main/dereverb_room_anvuew_sdr_13.7432.ckpt'],
    # === Karaoke Models ===
    'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt': ['https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml', 'https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'],
    'Karaoke_GaboxV1.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/karaoke/karaokegabox_1750911344.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/karaoke/Karaoke_GaboxV1.ckpt'],
    'bs_karaoke_gabox_IS.ckpt': ['https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/bsroformers/karaoke_bs_roformer.yaml', 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/bsroformers/bs_karaoke_gabox_IS.ckpt'],
    'bs_roformer_karaoke_frazer_becruily.ckpt': ['https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/config_karaoke_frazer_becruily.yaml', 'https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/bs_roformer_karaoke_frazer_becruily.ckpt'],
    'mel_band_roformer_karaoke_becruily.ckpt': ['https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/config_karaoke_becruily.yaml', 'https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt'],
    # === Other / General Purpose Models ===
    'model_bs_roformer_ep_937_sdr_10.5309.ckpt': ['https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_937_sdr_10.5309.yaml', 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt'],
    'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt': ['https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml', 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'],
    'model_bandit_plus_dnr_sdr_11.47.chpt': ['https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml', 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt'],
    'checkpoint-multi_state_dict.ckpt': ['https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_dnr_bandit_v2_mus64.yaml', 'https://huggingface.co/jarredou/banditv2_state_dicts_only/resolve/main/checkpoint-multi_state_dict.ckpt'],
    'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt': ['https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml', 'https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'],
    'bs_hyperace.ckpt': [('https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/config.yaml', 'config_hyperace.yaml'), 'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/bs_hyperace.ckpt'],
    'becruily_deux.ckpt': ['https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/config_deux_becruily.yaml', 'https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt'],
    'becruily_guitar.ckpt': ['https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/config_guitar_becruily.yaml', 'https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt'],
    'aspiration_mel_band_roformer_sdr_18.9845.ckpt': ['https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/config_aspiration_mel_band_roformer.yaml', 'https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/aspiration_mel_band_roformer_sdr_18.9845.ckpt'],
    'model_mdx23c_ep_271_l1_freq_72.2383.ckpt': ['https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/config_mdx23c_similarity.yaml', 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/model_mdx23c_ep_271_l1_freq_72.2383.ckpt'],
    'model_mel_band_roformer_ep_72_sdr_3.2232.ckpt': ['https://huggingface.co/listra92/MyModels/resolve/main/misc/config_mel_band_roformer_Lead_Rhythm_Guitar.yaml', 'https://huggingface.co/listra92/MyModels/resolve/main/misc/model_mel_band_roformer_ep_72_sdr_3.2232.ckpt'],
    'last_bs_roformer.ckpt': ['https://huggingface.co/listra92/MyModels/resolve/main/misc/config.yaml', 'https://huggingface.co/listra92/MyModels/resolve/main/misc/last_bs_roformer.ckpt'],
    'bs_roformer_4stems_ft.pth': ['https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/config.yaml', 'https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/bs_roformer_4stems_ft.pth'],
    'checkpoint-eng_state_dict.ckpt': ['https://huggingface.co/jarredou/banditv2_state_dicts_only/resolve/main/config_dnr_bandit_v2_mus64.yaml', 'https://huggingface.co/jarredou/banditv2_state_dicts_only/resolve/main/checkpoint-eng_state_dict.ckpt'],
}

# Custom model URLs for models needing a custom .py file
MODEL_CUSTOM_PY_URLS = {
    'bs_hyperace.ckpt': 'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/bs_roformer.py',
}


def ensure_model_files_downloaded(checkpoint_filename, model_file_dir=None):
    """Pre-download all files for a model (checkpoint + config + optional .py) before loading.
    
    This must be called BEFORE separator.load_model() so the bypass in separator.py
    can find the model file locally and skip the registry check.
    
    Args:
        checkpoint_filename: The checkpoint filename (e.g. 'big_beta6x.ckpt')
        model_file_dir: Target directory (defaults to MODEL_CACHE_DIR)
    
    Returns: (success: bool, message: str)
    """
    target_dir = model_file_dir or MODEL_CACHE_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    urls = MODEL_DOWNLOAD_URLS.get(checkpoint_filename)
    if not urls:
        # Not in our URL registry — might be a base model handled by audio-separator
        logger.debug(f"No download URLs for {checkpoint_filename}, assuming base model")
        return True, "Base model (no pre-download needed)"
    
    for url_entry in urls:
        # Handle (url, target_filename) tuples for files that need renaming
        if isinstance(url_entry, tuple):
            url, target_name = url_entry
        else:
            url = url_entry
            target_name = os.path.basename(urllib.parse.urlparse(url).path.split('?')[0])
        
        success, result = download_model_from_url(url, target_name, target_dir)
        if not success:
            return False, f"Failed to download {target_name}: {result}"
    
    # Download custom .py if needed
    py_url = MODEL_CUSTOM_PY_URLS.get(checkpoint_filename)
    if py_url:
        py_name = os.path.basename(urllib.parse.urlparse(py_url).path.split('?')[0])
        success, result = download_model_from_url(py_url, py_name, target_dir)
        if not success:
            logger.warning(f"Failed to download custom .py: {result}")
    
    return True, f"All files downloaded for {checkpoint_filename}"


# ─── URL Helpers ─────────────────────────────────────────────────────────────

def fix_huggingface_url(url):
    """Auto-fix Hugging Face URLs to use /resolve/main/ for direct download."""
    if not url:
        return url
    url = url.strip()
    # Convert 'blob' to 'resolve' for HF URLs
    if 'huggingface.co' in url and '/blob/' in url:
        url = url.replace('/blob/', '/resolve/')
    return url


def download_model_from_url(url, target_filename=None, target_dir=None):
    """Download a model file from URL to the model cache directory.
    
    Args:
        url: Direct download URL
        target_filename: Optional filename (auto-detected from URL if not provided)
        target_dir: Target directory (defaults to MODEL_CACHE_DIR)
    
    Returns: (success: bool, file_path: str or error message)
    """
    if not url or not url.strip():
        return False, "URL is required"
    
    url = fix_huggingface_url(url.strip())
    save_dir = target_dir or MODEL_CACHE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    if not target_filename:
        parsed = urllib.parse.urlparse(url)
        target_filename = os.path.basename(parsed.path.split('?')[0])
    
    file_path = os.path.join(save_dir, target_filename)
    
    # Skip if already exists
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        logger.info(f"Model file already exists: {file_path}")
        return True, file_path
    
    try:
        logger.info(f"Downloading: {url} → {file_path}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"✅ Downloaded: {target_filename} ({size_mb:.1f} MB)")
            return True, file_path
        else:
            return False, f"Download failed: file is empty"
    except Exception as e:
        # Clean up partial download
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return False, f"Download error: {str(e)}"


# ─── Custom Model Management ────────────────────────────────────────────────

def load_custom_models():
    """Load custom models from JSON file.
    
    Custom models are stored as:
    {
        "Model Name": {
            "checkpoint_url": "https://...",
            "config_url": "https://...",     # optional
            "custom_model_url": "https://...",  # optional .py file
            "checkpoint_filename": "model.ckpt"
        }
    }
    """
    if os.path.exists(CUSTOM_MODELS_FILE):
        try:
            with open(CUSTOM_MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading custom models: {e}")
    return {}


def save_custom_models(models):
    """Save custom models to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(CUSTOM_MODELS_FILE)), exist_ok=True)
    with open(CUSTOM_MODELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(models, f, indent=2, ensure_ascii=False)


def add_custom_model(model_name, checkpoint_url, config_url=None, custom_model_url=None):
    """Add a custom model with download URLs.
    
    Args:
        model_name: Display name for the model
        checkpoint_url: Direct URL to the .ckpt/.pth checkpoint file
        config_url: Optional URL to the config .yaml file
        custom_model_url: Optional URL to a custom .py model file
    
    Returns: (success: bool, message: str)
    """
    if not model_name or not model_name.strip():
        return False, "Model name is required"
    if not checkpoint_url or not checkpoint_url.strip():
        return False, "Checkpoint URL is required"
    
    model_name = model_name.strip()
    checkpoint_url = fix_huggingface_url(checkpoint_url.strip())
    config_url = fix_huggingface_url(config_url.strip()) if config_url and config_url.strip() else None
    custom_model_url = fix_huggingface_url(custom_model_url.strip()) if custom_model_url and custom_model_url.strip() else None
    
    # Extract checkpoint filename from URL
    parsed = urllib.parse.urlparse(checkpoint_url)
    checkpoint_filename = os.path.basename(parsed.path.split('?')[0])
    if not checkpoint_filename:
        return False, "Could not extract filename from checkpoint URL"
    
    models = load_custom_models()
    if model_name in models:
        return False, f"Model '{model_name}' already exists"
    
    models[model_name] = {
        'checkpoint_url': checkpoint_url,
        'config_url': config_url,
        'custom_model_url': custom_model_url,
        'checkpoint_filename': checkpoint_filename,
    }
    save_custom_models(models)
    return True, f"✅ Model '{model_name}' added successfully"


def delete_custom_model(model_name):
    """Delete a custom model and optionally its cached files.
    
    Returns: (success: bool, message: str)
    """
    models = load_custom_models()
    if model_name not in models:
        return False, f"Model '{model_name}' not found"
    
    # Try to clean up downloaded files
    model_config = models[model_name]
    if isinstance(model_config, dict):
        ckpt_file = os.path.join(MODEL_CACHE_DIR, model_config.get('checkpoint_filename', ''))
        if os.path.exists(ckpt_file):
            try:
                os.remove(ckpt_file)
                logger.info(f"Deleted cached model file: {ckpt_file}")
            except Exception as e:
                logger.warning(f"Could not delete cached file: {e}")
    
    del models[model_name]
    save_custom_models(models)
    return True, f"✅ Model '{model_name}' deleted"


def get_custom_models_list():
    """Get list of custom models as [(name, url)] tuples."""
    models = load_custom_models()
    result = []
    for name, config in models.items():
        if isinstance(config, dict):
            result.append((name, config.get('checkpoint_url', '')))
        else:
            result.append((name, str(config)))
    return result


def ensure_custom_model_downloaded(model_name):
    """Download custom model files if not already in cache.
    
    Returns: (success: bool, checkpoint_filename: str or error)
    """
    models = load_custom_models()
    if model_name not in models:
        return False, f"Custom model '{model_name}' not found"
    
    config = models[model_name]
    if not isinstance(config, dict):
        # Legacy format: just a filename string
        return True, str(config)
    
    # Download checkpoint
    ckpt_url = config.get('checkpoint_url')
    ckpt_filename = config.get('checkpoint_filename')
    if ckpt_url:
        success, result = download_model_from_url(ckpt_url, ckpt_filename)
        if not success:
            return False, f"Failed to download checkpoint: {result}"
    
    # Download config if provided
    config_url = config.get('config_url')
    if config_url:
        config_filename = f"config_{model_name.replace(' ', '_').lower()}.yaml"
        success, result = download_model_from_url(config_url, config_filename)
        if not success:
            logger.warning(f"Failed to download config: {result}")
    
    # Download custom .py if provided
    py_url = config.get('custom_model_url')
    if py_url:
        py_filename = os.path.basename(urllib.parse.urlparse(py_url).path.split('?')[0])
        success, result = download_model_from_url(py_url, py_filename)
        if not success:
            logger.warning(f"Failed to download custom .py: {result}")
    
    return True, ckpt_filename


def get_all_models():
    """Get EXTENDED_MODELS merged with custom models under 'Custom Models' category.
    
    Custom models are shown as {name: checkpoint_filename} for dropdown compatibility.
    """
    all_models = dict(EXTENDED_MODELS)
    custom = load_custom_models()
    if custom:
        # Flatten custom models to {name: checkpoint_filename} for UI dropdowns
        custom_flat = {}
        for name, config in custom.items():
            if isinstance(config, dict):
                custom_flat[name] = config.get('checkpoint_filename', '')
            else:
                custom_flat[name] = str(config)
        all_models["Custom Models"] = custom_flat
    return all_models


def get_model_choices(category):
    """Get model choices for a given category."""
    all_models = get_all_models()
    return list(all_models.get(category, {}).keys())


def get_categories():
    """Get all available categories."""
    return list(get_all_models().keys())


def find_model_filename(model_key):
    """Find checkpoint filename for a given model display name.
    
    For custom models, also ensures the file is downloaded to cache.
    """
    # Check built-in models first
    for category, models in EXTENDED_MODELS.items():
        if model_key in models:
            return models[model_key]
    
    # Check custom models
    custom = load_custom_models()
    if model_key in custom:
        config = custom[model_key]
        if isinstance(config, dict):
            # Auto-download from URL if needed
            success, result = ensure_custom_model_downloaded(model_key)
            if success:
                return result
            else:
                logger.error(f"Failed to download custom model '{model_key}': {result}")
                return None
        else:
            return str(config)
    
    return None


# ─── Audio Segmentation Helpers ──────────────────────────────────────────────

def get_audio_duration(file_path):
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get duration for {file_path}: {e}")
    return 0


def split_audio_segments(input_path, output_dir, segment_duration=SEGMENT_DURATION):
    """Split audio into segments of given duration using ffmpeg.
    
    Returns list of segment file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    total_duration = get_audio_duration(input_path)
    if total_duration <= 0:
        logger.error(f"Could not determine duration for {input_path}")
        return []
    
    segments = []
    seg_idx = 0
    start_time = 0
    
    while start_time < total_duration:
        seg_filename = f"{base_name}_seg{seg_idx:03d}.wav"
        seg_path = os.path.join(output_dir, seg_filename)
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2',
            seg_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and os.path.exists(seg_path):
                segments.append(seg_path)
                logger.info(f"✅ Segment {seg_idx}: {seg_filename}")
            else:
                logger.warning(f"⚠️ Failed to create segment {seg_idx}: {result.stderr[-200:]}")
        except Exception as e:
            logger.warning(f"⚠️ Error creating segment {seg_idx}: {e}")
        
        seg_idx += 1
        start_time += segment_duration
    
    return segments


def concatenate_audio_files(file_list, output_path, output_format='wav'):
    """Concatenate multiple audio files into one using ffmpeg concat.
    
    Returns: output file path or None on failure.
    """
    if not file_list:
        return None
    
    if len(file_list) == 1:
        shutil.copy(file_list[0], output_path)
        return output_path
    
    concat_list = output_path + '.concat.txt'
    try:
        with open(concat_list, 'w') as f:
            for fp in file_list:
                f.write(f"file '{fp}'\n")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_list, '-c', 'copy', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            logger.error(f"Concatenation failed: {result.stderr[-300:]}")
            return None
    except Exception as e:
        logger.error(f"Concatenation error: {e}")
        return None
    finally:
        if os.path.exists(concat_list):
            try:
                os.remove(concat_list)
            except:
                pass


def concatenate_segment_outputs(output_dir, output_format='wav'):
    """After processing segments, concatenate per-instrument outputs.
    
    Each segment produces files like: seg000_vocals.wav, seg000_instrumental.wav
    This function concatenates all vocal segments, all instrumental segments, etc.
    """
    # Order matters: check longer names first to avoid substring conflicts
    instrument_types = ['instrumental', 'instrument', 'phaseremix', 'vocals', 'drum',
                       'bass', 'other', 'effects', 'speech', 'music', 'dry',
                       'male', 'female', 'bleed', 'karaoke']
    
    output_files = sorted(os.listdir(output_dir))
    already_processed = set()
    
    for inst_type in instrument_types:
        # Regex exact suffix matching: _instrument. but NOT _instrumental.
        pattern = re.compile(r'_' + re.escape(inst_type) + r'\.', re.IGNORECASE)
        
        inst_files = sorted([
            os.path.join(output_dir, f) for f in output_files
            if pattern.search(f) and '_seg' in f.lower() and f not in already_processed
        ])
        
        if len(inst_files) <= 1:
            continue
        
        for f in inst_files:
            already_processed.add(os.path.basename(f))
        
        logger.info(f"🔗 Concatenating {len(inst_files)} {inst_type} segments...")
        
        # Determine output filename (remove _segXXX from first segment's name)
        first_name = os.path.basename(inst_files[0])
        concat_output_name = re.sub(r'_seg\d+', '', first_name)
        concat_output = os.path.join(output_dir, concat_output_name)
        
        result = concatenate_audio_files(inst_files, concat_output, output_format)
        
        if result:
            # Remove segment files
            for seg_file in inst_files:
                try:
                    os.remove(seg_file)
                except:
                    pass
            logger.info(f"✅ Concatenated {inst_type}: {os.path.basename(concat_output)}")
        else:
            logger.warning(f"⚠️ Concatenation failed for {inst_type}")
