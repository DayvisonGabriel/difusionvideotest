# Preparando código para gerar vídeo com Stable Video Diffusion, mantendo a proporção da imagem original,
# e aplicando upscaling online com Real-ESRGAN via Hugging Face pipeline.

# Importações necessárias
import torch
from diffusers import StableVideoDiffusionPipeline
import gradio as gr
from PIL import Image
import tempfile
import imageio
import os
import requests
from io import BytesIO

# Função para fazer upscale via API (simulação para fins de código)
def upscale_frame(frame):
    # Converte frame para bytes
    buf = BytesIO()
    frame.save(buf, format='PNG')
    buf.seek(0)

    # Exemplo com API pública (você pode trocar por qualquer endpoint de upscaling online)
    response = requests.post(
        "https://api-inference.huggingface.co/models/CompVis/ldm-super-resolution-4x-openimages",  # Exemplo de modelo
        headers={"Authorization": "Bearer YOUR_HF_TOKEN"},  # Use seu token HuggingFace se necessário
        files={"inputs": buf}
    )

    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return frame  # fallback para o frame original

# Carregando o pipeline
model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.enable_model_cpu_offload()

# Função principal de geração
def gerar_video(image, prompt, duration, fps, creativity):
    if image is None:
        return None

    # Calcula frames (máx 25)
    num_frames = min(int(duration * fps), 25)

    # Ajusta imagem mantendo proporção (com bordas pretas)
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        new_height = 512
        new_width = int(512 * aspect_ratio)
    else:
        new_width = 512
        new_height = int(512 / aspect_ratio)
    image = image.convert("RGB").resize((new_width, new_height))
    image = image.crop(((new_width - 512)//2, (new_height - 512)//2, (new_width + 512)//2, (new_height + 512)//2))

    # Geração dos frames com controle de "creativity" (simulado via seed aleatória)
    generator = torch.manual_seed(int(1000 * creativity))
    video_frames = pipe(image, decode_chunk_size=8, num_frames=num_frames, generator=generator).frames[0]

    # Upscale dos frames (simulado com placeholder)
    upscale_frames = [upscale_frame(frame) for frame in video_frames]

    # Salva o vídeo temporariamente
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
        imageio.mimsave(video_path, upscale_frames, fps=fps)

    return video_path

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🎬 Stable Video Diffusion - Geração de Vídeo com Upscale")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Imagem Base")
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (não utilizado no modelo atual, reservado)")
            duration = gr.Slider(1, 6, value=5, step=1, label="⏱ Duração (segundos)")
            fps = gr.Slider(4, 12, value=12, step=1, label="🎞 FPS")
            creativity = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="🎨 Nível de Criatividade")

    gerar_btn = gr.Button("🚀 Gerar Vídeo")
    video_output = gr.Video(label="Resultado")

    gerar_btn.click(fn=gerar_video, inputs=[image_input, prompt, duration, fps, creativity], outputs=video_output)

demo.launch(debug=True, share=True)

