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

# Variável global para armazenar o caminho do último vídeo
ultimo_video_path = None

# Redimensionamento com fundo preto para 951x537
def redimensionar_com_fundo_preto(img, alvo_largura=951, alvo_altura=537):
    img.thumbnail((alvo_largura, alvo_altura), Image.LANCZOS)
    fundo = Image.new("RGB", (alvo_largura, alvo_altura), (0, 0, 0))
    offset_x = (alvo_largura - img.width) // 2
    offset_y = (alvo_altura - img.height) // 2
    fundo.paste(img, (offset_x, offset_y))
    return fundo

# Função para apagar vídeo anterior
def limpar_video_anterior(nova_imagem):
    global ultimo_video_path
    if ultimo_video_path and os.path.exists(ultimo_video_path):
        os.remove(ultimo_video_path)
        print(f"🧹 Vídeo anterior removido: {ultimo_video_path}")
        ultimo_video_path = None
    return None  # Isso garante que o componente de vídeo seja limpo visualmente

# Função de geração de vídeo
def gerar_video(image, prompt, duration, fps, progress=gr.Progress(track_tqdm=True)):
    global ultimo_video_path

    if image is None:
        return None

    processed_image = redimensionar_com_fundo_preto(image)

    num_blocks = duration // 2
    frames_per_block = min(int(2 * fps), 25)

    current_input = processed_image
    all_frames = []

    for i in progress.tqdm(range(num_blocks), desc="Gerando vídeo"):
        result = pipe(current_input, decode_chunk_size=8, num_frames=frames_per_block).frames[0]
        all_frames.extend(result)
        current_input = result[-1]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
        imageio.mimsave(video_path, all_frames, fps=fps)

    ultimo_video_path = video_path
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

    # Quando uma nova imagem é enviada, limpa o vídeo
    image_input.upload(fn=limpar_video_anterior, inputs=image_input, outputs=video_output)

    gerar_btn.click(
        fn=gerar_video,
        inputs=[image_input, prompt, duration, fps],
        outputs=video_output
    )

demo.launch(debug=True, share=True)
