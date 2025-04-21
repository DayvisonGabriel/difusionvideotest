import torch
from diffusers import StableVideoDiffusionPipeline
import gradio as gr
from PIL import Image
import tempfile
import imageio
import os
from tqdm import tqdm

# Carrega o modelo
model_id = "stabilityai/stable-video-diffusion-img2vid-xt"

pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_model_cpu_offload()

# Redimensionamento com fundo preto para 951x537
def redimensionar_com_fundo_preto(img, alvo_largura=951, alvo_altura=537):
    img.thumbnail((alvo_largura, alvo_altura), Image.LANCZOS)
    fundo = Image.new("RGB", (alvo_largura, alvo_altura), (0, 0, 0))
    offset_x = (alvo_largura - img.width) // 2
    offset_y = (alvo_altura - img.height) // 2
    fundo.paste(img, (offset_x, offset_y))
    return fundo

# Função de geração
def gerar_video(image, prompt, duration, fps):
    if image is None:
        return None

    with gr.Progress(track_tqdm=True):
        # Ajusta a imagem de entrada para o tamanho esperado
        processed_image = redimensionar_com_fundo_preto(image)

        # Calcula blocos de 2s e frames por bloco
        num_blocks = duration // 2
        frames_per_block = min(int(2 * fps), 25)  # máximo 25

        current_input = processed_image
        all_frames = []

        for i in tqdm(range(num_blocks), desc="Gerando vídeo"):
            result = pipe(current_input, decode_chunk_size=8, num_frames=frames_per_block).frames[0]
            all_frames.extend(result)
            current_input = result[-1]  # último frame vira a próxima entrada

        # Salva vídeo em arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
            imageio.mimsave(video_path, all_frames, fps=fps)

    return video_path

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Stable Video Diffusion - Geração de Vídeo a partir de Imagem")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Imagem Base")
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (opcional)", placeholder="ex: estilo anime, realista, etc.")
            duration = gr.Slider(2, 12, value=4, step=2, label="⏱️ Duração do Vídeo (segundos)")
            fps = gr.Slider(4, 12, value=12, step=1, label="🎞️ FPS")

    gerar_btn = gr.Button("🚀 Gerar Vídeo")
    video_output = gr.Video(label="🎬 Resultado do Vídeo")

    gerar_btn.click(fn=gerar_video, inputs=[image_input, prompt, duration, fps], outputs=video_output)

demo.launch(debug=True, share=True)
