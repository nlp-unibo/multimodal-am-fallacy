from transformers import Wav2Vec2Processor, TFWav2Vec2Model, AutoProcessor, ClapAudioModel,  ClapModel, ClapProcessor
import torch
import numpy as np
import laion_clap
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)



def audio_processor(snippet_audio, audio_model='wav2vec', audio_model_sample_rate=16000):
    if audio_model == 'wav2vec':
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        input_values = processor(snippet_audio, sampling_rate=audio_model_sample_rate, return_tensors="tf").input_values[0]
        audio_emb = model(snippet_audio[None, :]).last_hidden_state
        #print(audio_emb.shape)
        return audio_emb

    elif audio_model == 'clap':
        # processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused") 
        # model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
        # model.get_embedd
        # inputs = processor(audios=snippet_audio, sampling_rate=audio_model_sample_rate, return_tensors="tf")
        # outputs = model(**inputs)
        # audio_embeds = outputs.last_hidden_state
        # print(audio_embeds.shape)
        
        # return audio_embeds

        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt() # download the default pretrained checkpoint.
        audio_data = snippet_audio.reshape(1, -1) # Make it (1,T) or (N,T)
        #audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
        audio_embed = model.get_audio_embedding_from_data(x = audio_data)
        #print(audio_embed.shape)

        # model = ClapModel.from_pretrained("laion/clap-htsat-fused")
        # processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

        # audio_data = snippet_audio.reshape(1, -1) # Make it (1,T) or (N,T)
        # audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() 
        # inputs = processor(audios=audio_data, return_tensors="tf")
        # audio_embed = model.get_audio_features(**inputs)
        # print(audio_embed.shape)
        #audio_embed = audio_embed.cpu().data.numpy()
        #print(audio_embed.shape)

        return audio_embed








