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

    # Garante que duration é múltiplo de 2 (já forçado pelo slider)
    num_blocks = duration // 2
    frames_per_block = min(int(2 * fps), 25)  # máximo de 25 frames por bloco

    # Redimensiona proporcionalmente para caber em 512x512 com barras pretas
    original_size = image.size
    image.thumbnail((512, 512), Image.LANCZOS)
    thumb_size = image.size

    background = Image.new("RGB", (512, 512), (0, 0, 0))
    offset = ((512 - thumb_size[0]) // 2, (512 - thumb_size[1]) // 2)
    background.paste(image, offset)

    current_input = background
    all_frames = []

    for i in range(num_blocks):
        result = pipe(current_input, decode_chunk_size=8, num_frames=frames_per_block).frames[0]
        all_frames.extend(result)

        # Pega o último frame do bloco e usa como próxima entrada
        last_frame = result[-1]
        current_input = last_frame

    # Recorte proporcional do vídeo final
    crop_width, crop_height = thumb_size
    crop_x = (512 - crop_width) // 2
    crop_y = (512 - crop_height) // 2

    cropped_frames = [
        frame.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)) for frame in all_frames
    ]

    # Salva vídeo final
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
            duration = gr.Slider(2, 12, value=4, step=2, label="Duração (segundos)")  # até 12s
            fps = gr.Slider(4, 12, value=12, step=1, label="FPS")

    gerar_btn = gr.Button("Gerar Vídeo")
    video_output = gr.Video(label="Resultado")

    gerar_btn.click(fn=gerar_video, inputs=[image_input, prompt, duration, fps], outputs=video_output)

demo.launch(debug=True, share=True)


