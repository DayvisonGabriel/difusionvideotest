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

# Função para redimensionar a imagem proporcionalmente e adicionar fundo preto
def redimensionar_imagem(imagem):
    largura, altura = imagem.size
    
    # Calculando a proporção de redimensionamento
    proporcao = min(512 / largura, 512 / altura)
    
    # Calculando as novas dimensões
    nova_largura = int(largura * proporcao)
    nova_altura = int(altura * proporcao)
    
    # Redimensionando a imagem
    imagem_redimensionada = imagem.resize((nova_largura, nova_altura))
    
    # Criando uma nova imagem de fundo preto com 512x512
    imagem_com_fundo = Image.new("RGB", (512, 512), (0, 0, 0))  # Cor preta (0, 0, 0)
    
    # Calculando a posição de inserção da imagem redimensionada no centro
    x_offset = (512 - nova_largura) // 2
    y_offset = (512 - nova_altura) // 2
    
    # Colocando a imagem redimensionada sobre o fundo preto
    imagem_com_fundo.paste(imagem_redimensionada, (x_offset, y_offset))
    
    return imagem_com_fundo

# Função de geração do vídeo
def gerar_video(image, prompt, duration, fps, progress=gr.Progress()):
    if image is None:
        return None

    # Garante que duration é múltiplo de 2 (já forçado pelo slider)
    num_blocks = duration // 2
    frames_per_block = min(int(2 * fps), 25)  # máximo de 25 frames por bloco

    # Processamento da imagem de entrada com redimensionamento e fundo preto
    processed_image = redimensionar_imagem(image)

    current_input = processed_image
    all_frames = []

    # Atualizando progresso inicial
    total_frames = num_blocks * frames_per_block
    progress(0, total_frames)  # Começa do zero e vai até o total de frames

    for i in range(num_blocks):
        result = pipe(current_input, decode_chunk_size=8, num_frames=frames_per_block).frames[0]
        all_frames.extend(result)

        # Usa o último frame como nova entrada
        last_frame = result[-1]
        current_input = last_frame

        # Atualiza o progresso durante o processamento
        frames_processed = (i + 1) * frames_per_block
        progress(frames_processed, total_frames)

    # Salva vídeo final
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
        imageio.mimsave(video_path, all_frames, fps=fps)

    return video_path

# Interface
with gr.Blocks() as demo:
    gr.Markdown("## Stable Video Diffusion - Vídeo a partir de Imagem")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Imagem Base")
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (não utilizado no modelo atual, mas reservado)", placeholder="ex: estilo Studio Ghibli")
            duration = gr.Slider(2, 12, value=4, step=2, label="Duração (segundos)")  # até 12s, múltiplos de 2
            fps = gr.Slider(4, 12, value=12, step=1, label="FPS")

    gerar_btn = gr.Button("Gerar Vídeo")
    video_output = gr.Video(label="Resultado")
    progress_bar = gr.Progress()

    gerar_btn.click(fn=gerar_video, inputs=[image_input, prompt, duration, fps, progress_bar], outputs=video_output)

demo.launch(debug=True, share=True)
