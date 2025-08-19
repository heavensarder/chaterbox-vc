import random
import numpy as np
import torch
import gradio as gr
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_tts_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


def load_vc_model():
    model = ChatterboxVC.from_pretrained(DEVICE)
    return model


def tts_generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    output_path = "generated_tts.wav"
    ta.save(output_path, wav, model.sr)
    return (model.sr, wav.squeeze(0).numpy()), output_path


def vc_generate(model, audio, target_voice_path):
    if model is None:
        model = ChatterboxVC.from_pretrained(DEVICE)

    wav = model.generate(
        audio, target_voice_path=target_voice_path,
    )
    output_path = "generated_vc.wav"
    ta.save(output_path, wav, model.sr)
    return (model.sr, wav.squeeze(0).numpy()), output_path


with gr.Blocks(theme=gr.themes.Base(primary_hue="green", secondary_hue="blue")) as demo:
    gr.Markdown("# Chatterbox: Text-to-Speech and Voice Conversion")
    tts_model_state = gr.State(None)
    vc_model_state = gr.State(None)

    with gr.Tabs():
        with gr.TabItem("Text-to-Speech"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        tts_text = gr.Textbox(
                            value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                            label="Text to synthesize (max chars 300)",
                            max_lines=5
                        )
                        tts_ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
                    
                    with gr.Group():
                        with gr.Row():
                            tts_exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration", value=.5)
                            tts_cfg_weight = gr.Slider(0.0, 1, step=.05, label="Pace", value=0.5)

                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Group():
                            with gr.Row():
                                tts_temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)
                                tts_repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="Repetition Penalty", value=1.2)
                            with gr.Row():
                                tts_min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P", value=0.05)
                                tts_top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=1.00)
                            tts_seed_num = gr.Number(value=0, label="Random seed (0 for random)")

                    tts_run_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=1):
                    tts_audio_output = gr.Audio(label="Output Audio")
                    tts_file_output = gr.File(label="Download")

        with gr.TabItem("Voice Conversion"):
            with gr.Row():
                with gr.Column():
                    vc_audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file")
                    vc_target_voice = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file", value=None)
                    vc_run_btn = gr.Button("Convert", variant="primary")
                with gr.Column():
                    vc_audio_output = gr.Audio(label="Output Audio")
                    vc_file_output = gr.File(label="Download")

    demo.load(fn=load_tts_model, inputs=[], outputs=tts_model_state)
    demo.load(fn=load_vc_model, inputs=[], outputs=vc_model_state)

    tts_run_btn.click(
        fn=tts_generate,
        inputs=[
            tts_model_state,
            tts_text,
            tts_ref_wav,
            tts_exaggeration,
            tts_temp,
            tts_seed_num,
            tts_cfg_weight,
            tts_min_p,
            tts_top_p,
            tts_repetition_penalty,
        ],
        outputs=[tts_audio_output, tts_file_output],
    )

    vc_run_btn.click(
        fn=vc_generate,
        inputs=[
            vc_model_state,
            vc_audio_in,
            vc_target_voice,
        ],
        outputs=[vc_audio_output, vc_file_output],
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True, show_api=False)
