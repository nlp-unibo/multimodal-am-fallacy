from transformers import Wav2Vec2Processor, TFWav2Vec2Model


def audio_processor(snippet_audio, audio_model='wav2vec', audio_model_sampling_rate=16000):
    if audio_model == 'wav2vec':
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        input_values = processor(snippet_audio, sampling_rate=audio_model_sampling_rate, return_tensors="tf").input_values
        return model(input_values)

