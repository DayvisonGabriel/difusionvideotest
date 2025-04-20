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

    num_frames = min(int(duration * fps), 25)  # modelo suporta até 25 frames

    # Guarda o tamanho proporcional original
    original_size = image.size  # (width, height)

    # Redimensiona proporcionalmente para caber em 512x512
    image.thumbnail((512, 512), Image.LANCZOS)  # mantém proporção
    thumb_size = image.size

    # Cria fundo preto 512x512
    background = Image.new("RGB", (512, 512), (0, 0, 0))
    offset = ((512 - thumb_size[0]) // 2, (512 - thumb_size[1]) // 2)
    background.paste(image, offset)
    
    # Usa imagem com barras pretas como input para o modelo
    video_frames = pipe(background, decode_chunk_size=8, num_frames=num_frames).frames[0]

    # Recorta os frames para o tamanho proporcional original (sem as barras pretas)
    crop_width, crop_height = thumb_size
    crop_x = (512 - crop_width) // 2
    crop_y = (512 - crop_height) // 2

    cropped_frames = [frame.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)) for frame in video_frames]

    # Salva vídeo temporário
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
        imageio.mimsave(video_path, cropped_frames, fps=fps)

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
