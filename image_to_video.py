import torch
from diffusers import StableVideoDiffusionPipeline
import gradio as gr
from PIL import Image
import tempfile
import imageio
import os

# Carrega o modelo
model_id = "stabilityai/stable-video-diffusion-img2vid-xt"

pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_model_cpu_offload()

# Função de geração
def gerar_video(image, prompt, duration, fps):
    if image is None:
        return None

    # Guardar resolução original
    original_size = image.size  # (width, height)

    num_frames = min(int(duration * fps), 25)  # máximo 25 frames

    # Redimensionar para 512x512 para o modelo
    input_image = image.convert("RGB").resize((512, 512))

    # Geração com o modelo
    video_frames = pipe(input_image, decode_chunk_size=8, num_frames=num_frames).frames[0]

    # Redimensionar frames para o tamanho original da imagem
    resized_frames = [frame.resize(original_size, Image.LANCZOS) for frame in video_frames]

    # Salvar vídeo
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
        imageio.mimsave(video_path, resized_frames, fps=fps)

    return video_path

# Interface
with gr.Blocks() as demo:
    gr.Markdown("## Stable Video Diffusion - Vídeo a partir de Imagem")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Imagem Base")
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (não utilizado no modelo atual, mas reservado)", placeholder="ex: estilo Studio Ghibli")
            duration = gr.Slider(1, 6, value=5, step=1, label="Duração (segundos)")
            fps = gr.Slider(4, 12, value=12, step=1, label="FPS")

    gerar_btn = gr.Button("Gerar Vídeo")
    video_output = gr.Video(label="Resultado")

    gerar_btn.click(fn=gerar_video, inputs=[image_input, prompt, duration, fps], outputs=video_output)

demo.launch(debug=True, share=True)
